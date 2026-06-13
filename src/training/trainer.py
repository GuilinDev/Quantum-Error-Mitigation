"""Training utilities for error mitigation models.

This module provides the training loop, configuration, and
experiment tracking integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_type: str = "standard"
    circuit_dim: int = 50
    noise_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # 'cosine' or 'onecycle'

    # Loss
    loss_type: str = "mse"  # 'mse', 'huber', or 'combined'
    huber_delta: float = 0.1

    # Regularization
    dropout: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    use_wandb: bool = True
    project_name: str = "sqai2026-error-mitigation"
    run_name: Optional[str] = None

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v if not isinstance(v, list) else str(v)
            for k, v in self.__dict__.items()
        }


class Trainer:
    """Trainer for error mitigation models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Initialize trainer.

        Args:
            model: Neural network model.
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Set up optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Set up scheduler
        total_steps = len(train_loader) * config.num_epochs
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=config.learning_rate / 100,
            )
        else:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=config.warmup_epochs / config.num_epochs,
            )

        # Set up loss function
        self.loss_fn = self._get_loss_fn()

        # Tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B if available and enabled
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.to_dict(),
            )
            wandb.watch(model)

    def _get_loss_fn(self) -> Callable:
        """Get loss function based on config."""
        if self.config.loss_type == "mse":
            return nn.MSELoss()
        elif self.config.loss_type == "huber":
            return nn.HuberLoss(delta=self.config.huber_delta)
        elif self.config.loss_type == "combined":
            mse = nn.MSELoss()
            huber = nn.HuberLoss(delta=self.config.huber_delta)
            return lambda pred, target: 0.5 * mse(pred, target) + 0.5 * huber(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move to device
            circuit_features = batch["circuit_features"].to(self.device)
            noise_features = batch["noise_features"].to(self.device)
            noisy_values = batch["noisy_value"].to(self.device)
            ideal_values = batch["ideal_value"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            mitigated = self.model(noisy_values, circuit_features, noise_features)

            # Compute loss
            loss = self.loss_fn(mitigated, ideal_values)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_improvement = 0.0
        num_batches = 0

        for batch in self.val_loader:
            circuit_features = batch["circuit_features"].to(self.device)
            noise_features = batch["noise_features"].to(self.device)
            noisy_values = batch["noisy_value"].to(self.device)
            ideal_values = batch["ideal_value"].to(self.device)

            # Forward pass
            mitigated = self.model(noisy_values, circuit_features, noise_features)

            # Compute metrics
            loss = self.loss_fn(mitigated, ideal_values)
            mae = torch.abs(mitigated - ideal_values).mean()

            # Compute improvement over noisy
            noisy_error = torch.abs(noisy_values - ideal_values).mean()
            mitigated_error = torch.abs(mitigated - ideal_values).mean()
            improvement = (noisy_error - mitigated_error) / (noisy_error + 1e-8)

            total_loss += loss.item()
            total_mae += mae.item()
            total_improvement += improvement.item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_mae": total_mae / num_batches,
            "val_improvement": total_improvement / num_batches,
        }

    def train(self) -> Dict[str, List[float]]:
        """Run full training.

        Returns:
            Dictionary of training history.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics["val_loss"])

            epoch_time = time.time() - start_time

            # Print progress
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['val_loss']:.6f} | "
                f"Improvement: {val_metrics['val_improvement']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Log to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        **val_metrics,
                    }
                )

            # Save checkpoint if best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        if self.use_wandb:
            wandb.finish()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "config": self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]


class EvaluationMetrics:
    """Evaluation metrics for error mitigation."""

    @staticmethod
    def mean_absolute_error(
        predicted: np.ndarray, target: np.ndarray
    ) -> float:
        """Compute mean absolute error."""
        return np.abs(predicted - target).mean()

    @staticmethod
    def improvement_ratio(
        mitigated: np.ndarray,
        noisy: np.ndarray,
        ideal: np.ndarray,
    ) -> float:
        """Compute relative improvement over noisy baseline.

        Returns:
            Fraction of error reduced by mitigation.
        """
        noisy_error = np.abs(noisy - ideal).mean()
        mitigated_error = np.abs(mitigated - ideal).mean()
        return (noisy_error - mitigated_error) / (noisy_error + 1e-8)

    @staticmethod
    def energy_accuracy(
        predicted_energy: float,
        exact_energy: float,
        chemical_accuracy: float = 0.0016,  # ~1 kcal/mol in Hartree
    ) -> bool:
        """Check if energy is within chemical accuracy."""
        return abs(predicted_energy - exact_energy) < chemical_accuracy

    @staticmethod
    def approximation_ratio(
        achieved_cost: float,
        optimal_cost: float,
    ) -> float:
        """Compute QAOA approximation ratio."""
        if optimal_cost == 0:
            return 1.0 if achieved_cost == 0 else 0.0
        return achieved_cost / optimal_cost
