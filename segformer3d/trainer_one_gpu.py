import os
import torch
import wandb
from tqdm import tqdm
from typing import Dict
from termcolor import colored
from torch.utils.data import DataLoader

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
        device: torch.device,
        save_path: str,
        wandb: bool,
    ):
        self.config = config

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.training_scheduler = training_scheduler
        self.num_epochs = num_epochs

        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_dice = 0.0

        self.compute_metrics = compute_metrics
        self.metrics_fn = metrics_fn

        self.log_interval = log_interval
        self.wandb = wandb

        self.save_path = save_path

    def _train_step(self):
        epoch_avg_loss = 0.0
        self.model.train()

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            unit="batch",
        ) as pbar:
            for idx, raw_data in enumerate(self.train_loader):
                data = raw_data["image"].to(self.device)
                labels = raw_data["label"].to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(data)
                loss = self.criterion(preds, labels)

                loss.backward()

                self.optimizer.step()

                epoch_avg_loss += loss.item()
                pbar.set_postfix({"train_loss": loss.item() / (idx + 1)})
                pbar.update(1)

            epoch_avg_loss /= len(self.train_loader)

            self.epoch_train_loss = epoch_avg_loss

        # return epoch_avg_loss

    def _val_step(self):
        epoch_avg_loss = 0.0
        total_dice = 0.0

        self.model.eval()

        with torch.no_grad():
            with tqdm(
                total=len(self.valid_loader), desc="Validation", unit="batch"
            ) as pbar:
                for idx, raw_data in enumerate(self.valid_loader):
                    data = raw_data["image"].to(self.device)
                    labels = raw_data["label"].to(self.device)

                    preds = self.model(data)
                    loss = self.criterion(preds, labels)

                    epoch_avg_loss += loss.item()

                    if self.compute_metrics:
                        dice = self.metrics_fn(data, labels, model=self.model)
                        total_dice += dice

                    pbar.set_postfix(
                        {
                            "val_loss": loss.item() / (idx + 1),
                            "val_dice": total_dice / (idx + 1),
                        }
                    )
                    pbar.update(1)

                epoch_avg_loss /= len(self.valid_loader)

                if self.compute_metrics:
                    total_dice /= len(self.valid_loader)
                    self.epoch_val_dice = total_dice

                self.epoch_val_loss = epoch_avg_loss

        return epoch_avg_loss, total_dice

    def _run_train_val(self):

        if self.wandb:
            wandb.watch(self.model, criterion=self.criterion, log="all", log_freq=10)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(colored(f"Epoch {self.current_epoch + 1}/{self.num_epochs}", "green"))

            # Training
            self._train_step()

            # Validation
            val_loss, total_dice = self._val_step()

            # Update metrics

            save_checkpoint = self._update_metrics()
            if save_checkpoint:
                self._save_checkpoint()

            if self.wandb:
                self._log_metrics()

            self.training_scheduler.step(val_loss)

    def _update_metrics(self) -> bool:
        save_checkpoint = False
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss
            save_checkpoint = True

        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss
            save_checkpoint = True  # Save the best validation loss model

        if self.compute_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

        return save_checkpoint

    def _log_metrics(self) -> None:
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
            "learning_rate": self.optimizer.param_groups[0]['lr'],  # Log learning rate
        }
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
        print(colored("Training completed", "green"))
