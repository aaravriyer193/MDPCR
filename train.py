# MDPCR — training/trainer.py
# Training loop: backpropagates through cloud iterations.
# Loss = distance between converged cloud and target response cloud.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.mdpcr import MDPCRModel
from training.checkpoint import save_checkpoint, load_checkpoint
from utils.logging import TrainingLogger
import config
import os


class Trainer:
    """
    Trains the MDPCR model on (prompt cloud → response cloud) pairs.

    The gradient flows through every iteration of the cloud update,
    so the influence layer learns to produce transformations that
    move prompt clouds toward their target response clouds.
    """

    def __init__(
        self,
        model      : MDPCRModel,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        logger     : TrainingLogger | None = None,
    ):
        self.model        = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.logger       = logger or TrainingLogger()

        self.optimiser = torch.optim.AdamW(
            model.parameters(),
            lr           = config.LEARNING_RATE,
            weight_decay = 1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode     = "min",
            factor   = 0.5,
            patience = 3,
            min_lr   = config.LEARNING_RATE * 0.05,
        )

        self.best_val_loss = float("inf")
        self.epoch         = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            prompt_pos = batch["prompt_positions"].to(config.DEVICE)  # (B, M_p, N)
            prompt_den = batch["prompt_densities"].to(config.DEVICE)  # (B, M_p)
            target_pos = batch["target_positions"].to(config.DEVICE)  # (B, M_t, N)

            batch_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)

            # Process each example individually (clouds have different sizes)
            for i in range(prompt_pos.shape[0]):
                p_pos = prompt_pos[i]  # (M_p, N)
                p_den = prompt_den[i]  # (M_p,)
                t_pos = target_pos[i]  # (M_t, N)

                # Forward: run cloud to convergence (training mode = fixed steps)
                converged, tracker = self.model(p_pos, p_den, training=True)

                # Loss: distance between converged cloud and target cloud
                loss = self.model.loss(converged, t_pos)
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / prompt_pos.shape[0]

            self.optimiser.zero_grad()
            batch_loss.backward()

            # Gradient clipping for stability across many unrolled iterations
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimiser.step()

            total_loss += batch_loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            prompt_pos = batch["prompt_positions"].to(config.DEVICE)
            prompt_den = batch["prompt_densities"].to(config.DEVICE)
            target_pos = batch["target_positions"].to(config.DEVICE)

            for i in range(prompt_pos.shape[0]):
                converged, _ = self.model(
                    prompt_pos[i], prompt_den[i], training=False
                )
                loss = self.model.loss(converged, target_pos[i])
                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def train(self, resume_from: str | None = None) -> None:
        """Full training run."""
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        if resume_from:
            self.epoch = load_checkpoint(self.model, self.optimiser, resume_from)
            print(f"Resuming from epoch {self.epoch}")

        params = self.model.parameter_count()
        print(f"Model parameters: {params['trainable']:,} trainable / "
              f"{params['total']:,} total")
        print(f"Training on {config.DEVICE}")
        print(f"Epochs: {config.EPOCHS}, Batch size: {config.BATCH_SIZE}")
        print("─" * 60)

        for epoch in range(self.epoch, config.EPOCHS):
            self.epoch = epoch + 1

            train_loss = self.train_epoch()
            val_loss   = self.val_epoch()
            self.scheduler.step(val_loss)

            self.logger.log(epoch=self.epoch, train_loss=train_loss,
                            val_loss=val_loss)

            print(f"Epoch {self.epoch:3d}/{config.EPOCHS} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {self.optimiser.param_groups[0]['lr']:.2e}")

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimiser, self.epoch,
                    os.path.join(config.CHECKPOINT_DIR, "best.pt")
                )
                print(f"  ✓ New best val loss: {val_loss:.6f}")

            # Periodic checkpoint
            if self.epoch % 10 == 0:
                save_checkpoint(
                    self.model, self.optimiser, self.epoch,
                    os.path.join(config.CHECKPOINT_DIR, f"epoch_{self.epoch}.pt")
                )

        print("─" * 60)
        print(f"Training complete. Best val loss: {self.best_val_loss:.6f}")