import os
import torch
import wandb
from tqdm import tqdm
from typing import Dict
from termcolor import colored
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.segmentation_metrics import SlidingWindowInference

def ddp_setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend=backend, rank=rank, world_size=world_size)

class Brats2018_Trainer:
    def __init__(self,
                 config: Dict,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 warmup_scheduler: torch.optim.lr_scheduler.StepLR,
                 training_scheduler: torch.optim.lr_scheduler.StepLR,
                 gpu_id: int,
                 print_every: int,
                 wandb=False):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.print_every = print_every
        self.wandb = wandb
        
        # Initialize DistributedDataParallel if using multiple GPUs
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"]
        )
        
        # Metrics
        self.current_epoch = 0
        self.epoch_train_loss = 0.0
        self.best_train_loss = float('inf')
        self.epoch_val_loss = 0.0
        self.best_val_loss = float('inf')
        self.epoch_val_dice = 0.0
        self.best_val_dice = 0.0

    def _configure_trainer(self) -> None:
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.val_ema_every = self.config["ema"]["val_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"]["checkpoint_save_dir"]

    def _train_step(self):
        epoch_avg_loss = 0.0
        self.model.train()

        for idx, raw_data in enumerate(self.train_loader):
            data = raw_data["image"].to(self.device)
            labels = raw_data["label"].to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_avg_loss += loss.item()
            
            if idx % self.print_every == 0:
                print(f"Epoch {self.current_epoch}, Loss: {loss.item()}")

        epoch_avg_loss /= len(self.train_loader)
        self.epoch_train_loss = epoch_avg_loss
        
        return epoch_avg_loss

    def _val_step(self):
        epoch_avg_loss = 0.0
        total_dice = 0.0
        self.model.eval()

        with torch.no_grad():
            for idx, raw_data in enumerate(self.valid_loader):
                data = raw_data["image"].to(self.device)
                labels = raw_data["label"].to(self.device)
                
                preds = self.model(data)
                loss = self.loss_fn(preds, labels)
                epoch_avg_loss += loss.item()

                if self.calculate_metrics:
                    dice = self.sliding_window_inference(data, labels, self.model)
                    total_dice += dice
        
        epoch_avg_loss /= len(self.valid_loader)
        self.epoch_val_loss = epoch_avg_loss
        
        if self.calculate_metrics:
            mean_dice = self._calc_dice_metric(data, labels)
            total_dice += mean_dice
            self.epoch_val_dice = total_dice
        
        return epoch_avg_loss, total_dice

    def _calc_dice_metric(self, data, labels) -> float:
        avg_dice_score = self.sliding_window_inference(data, labels, self.model)
        return avg_dice_score

    def _run_train_val(self):
        if self.wandb:
            wandb.watch(self.model, criterion=self.loss_fn, log="all", log_freq=10)
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            train_loss = self._train_step()
            val_loss, val_dice = self._val_step()
            
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Dice: {val_dice}")
            
            self._update_metrics()
            self._log_metrics()
            
            self.training_scheduler.step()
            
            if self.warmup_enabled and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                
            if epoch % self.print_every == 0 and self.device.type == 'cuda' and torch.distributed.get_rank() == 0:
                self.save_checkpoint()
    
    def _update_metrics(self) -> None:
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

    def _log_metrics(self) -> None:
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
        }
        wandb.log(log_data)
    
    def save_checkpoint(self) -> None:
        checkpoint_path = os.path.join(self.checkpoint_save_dir, f"model_epoch_{self.current_epoch}.pth")
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.epoch_train_loss,
                "val_loss": self.epoch_val_loss,
                "mean_dice": self.epoch_val_dice,
            },
            checkpoint_path,
        )
    
    def train(self) -> None:
        self._run_train_val()
        destroy_process_group()
