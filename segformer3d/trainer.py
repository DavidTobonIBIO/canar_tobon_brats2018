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


class Brats2018Trainer:
    def __init__(
        self,
        config: Dict,
        num_epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        compute_metrics: bool,
        metrics_fn: SlidingWindowInference,
        training_scheduler: torch.optim.lr_scheduler.StepLR,
        log_interval: int,
        gpu_id: int,
        save_path: str,
        wandb: bool,
    ):

        # Config
        self.config = config

        # Data Loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Model
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        # Model parameters

        self.optimizer = optimizer
        self.criterion = criterion
        self.training_scheduler = training_scheduler
        self.num_epochs = num_epochs

        # Losses
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_dice = 0.0

        # Metrics
        self.compute_metrics = compute_metrics
        self.metrics_fn = metrics_fn
        # Logging
        self.print_every = log_interval
        self.wandb = wandb

        # Path
        self.save_path = save_path

    def _train_step(self):
        torch.cuda.synchronize()
        epoch_avg_loss = 0.0
        self.model.train()

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} Rank:{torch.distributed.get_rank()}",
            unit="batch",
        ) as pbar:
            for idx, raw_data in enumerate(self.train_loader):
                data = raw_data["image"].to(self.gpu_id)
                labels = raw_data["label"].to(self.gpu_id)

                self.optimizer.zero_grad()
                preds = self.model(data)
                loss = self.criterion(preds, labels)

                torch.cuda.synchronize()

                loss.backward()

                # Ensure CUDA operations are synchronized before stepping optimizer
                torch.cuda.synchronize()
                self.optimizer.step()

                epoch_avg_loss += loss.item()

                pbar.set_postfix({"train_loss": epoch_avg_loss})
                pbar.update(1)

            epoch_avg_loss /= len(self.train_loader)
            if epoch_avg_loss < self.best_train_loss:
                self.best_train_loss = epoch_avg_loss

            self.epoch_train_loss = epoch_avg_loss

        torch.cuda.synchronize()

        return epoch_avg_loss

    def _val_step(self):
        torch.cuda.synchronize()
        epoch_avg_loss = 0.0
        total_dice = 0.0

        self.model.eval()
        with torch.no_grad():
            with tqdm(
                total=len(self.valid_loader), desc=f"Validation", unit="batch"
            ) as pbar:
                for idx, raw_data in enumerate(self.valid_loader):
                    data = raw_data["image"].to(self.gpu_id)
                    labels = raw_data["label"].to(self.gpu_id)

                    preds = self.model(data)
                    loss = self.criterion(preds, labels)
                    epoch_avg_loss += loss.item()

                    if self.compute_metrics and torch.distributed.get_rank() == 0:
                        dice = self.metrics_fn(data, labels, model=self.model)
                        total_dice += dice

                    torch.cuda.synchronize()

                    pbar.set_postfix(
                        {"val_loss": epoch_avg_loss, "mean_dice": total_dice}
                    )
                    pbar.update(1)

                epoch_avg_loss /= len(self.valid_loader)
                self.epoch_val_loss = epoch_avg_loss

                if self.compute_metrics:
                    total_dice /= len(self.valid_loader)
                    self.epoch_val_dice = total_dice

                if epoch_avg_loss < self.best_val_loss:
                    self.best_val_loss = epoch_avg_loss

        torch.cuda.synchronize()

        return epoch_avg_loss, total_dice

    def _run_train_val(self):

        if self.wandb and torch.distributed.get_rank() == 0:
            wandb.watch(self.model, criterion=self.criterion, log="all", log_freq=10)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(colored(f"Epoch {self.current_epoch + 1}/{self.num_epochs}", "green"))

            # Training
            train_loss = self._train_step()

            # Validation
            val_loss, val_dice = self._val_step()

            # Update metrics

            self._update_metrics()
            if self.wandb:
                self._log_metrics()

            self._save_checkpoint()

            self.training_scheduler.step()

        torch.cuda.synchronize()

    def _update_metrics(self) -> None:
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.compute_metrics:
            if self.epoch_val_dfice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

    def _log_metrics(self) -> None:
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
        }
        if torch.distributed.get_rank() == 0:
            wandb.log(log_data)

    def _save_checkpoint(self) -> None:

        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.epoch_train_loss,
            },
            self.save_path,
        )

    def train(self):
        self._run_train_val()
        destroy_process_group()
