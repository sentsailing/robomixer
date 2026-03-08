"""Training loop for the PairwiseScorer model.

Multi-task training with three loss components:
- BCE for transition quality score (0-1)
- MSE for overlap beats prediction
- Cross-entropy for transition type classification
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as f
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from robomixer.models.transition import TransitionType
from robomixer.scoring.embedding import PairwiseScorer
from robomixer.training.dataset import TransitionDataset

# Transition type label mapping
TRANSITION_TYPE_TO_IDX = {t: i for i, t in enumerate(TransitionType)}
NUM_TRANSITION_TYPES = len(TransitionType)


def compute_loss(
    quality_pred: torch.Tensor,
    overlap_pred: torch.Tensor,
    type_logits: torch.Tensor,
    quality_target: torch.Tensor,
    overlap_target: torch.Tensor,
    type_target: torch.Tensor | None = None,
    *,
    quality_weight: float = 1.0,
    overlap_weight: float = 0.5,
    type_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted multi-task loss.

    Args:
        quality_pred: (B, 1) predicted quality scores.
        overlap_pred: (B, 1) predicted overlap beats.
        type_logits: (B, NUM_TRANSITION_TYPES) transition type logits.
        quality_target: (B,) target quality scores.
        overlap_target: (B,) target overlap beats.
        type_target: (B,) target transition type indices. If None, type loss is skipped.
        quality_weight: Weight for quality BCE loss.
        overlap_weight: Weight for overlap MSE loss.
        type_weight: Weight for transition type cross-entropy loss.

    Returns:
        (total_loss, component_dict) where component_dict has individual loss values.
    """
    quality_loss = f.binary_cross_entropy(quality_pred.squeeze(-1), quality_target)
    overlap_loss = f.mse_loss(overlap_pred.squeeze(-1), overlap_target)

    components = {
        "quality_loss": quality_loss.item(),
        "overlap_loss": overlap_loss.item(),
    }

    total = quality_weight * quality_loss + overlap_weight * overlap_loss

    if type_target is not None:
        type_loss = f.cross_entropy(type_logits, type_target)
        total = total + type_weight * type_loss
        components["type_loss"] = type_loss.item()

    components["total_loss"] = total.item()
    return total, components


def train_scorer(
    dataset: TransitionDataset,
    *,
    point_feature_dim: int = 40,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
) -> PairwiseScorer:
    """Train the PairwiseScorer model.

    Args:
        dataset: TransitionDataset of labeled transitions.
        point_feature_dim: Dimension of each transition point feature vector.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate for Adam.
        val_fraction: Fraction of data for validation.
        checkpoint_dir: Directory to save model checkpoints.
        device: torch device string. Auto-selects CUDA if available.

    Returns:
        Trained PairwiseScorer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train/val split
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PairwiseScorer(point_feature_dim=point_feature_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            exit_feat = batch["exit_features"].to(device)
            entry_feat = batch["entry_features"].to(device)
            quality_target = batch["quality"].to(device)
            overlap_target = batch["overlap_beats"].to(device)

            quality_pred, overlap_pred, type_logits = model(exit_feat, entry_feat)

            # No type labels in current dataset — skip type loss during training
            loss, _ = compute_loss(
                quality_pred,
                overlap_pred,
                type_logits,
                quality_target,
                overlap_target,
                type_target=None,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                exit_feat = batch["exit_features"].to(device)
                entry_feat = batch["entry_features"].to(device)
                quality_target = batch["quality"].to(device)
                overlap_target = batch["overlap_beats"].to(device)

                quality_pred, overlap_pred, type_logits = model(exit_feat, entry_feat)

                loss, components = compute_loss(
                    quality_pred,
                    overlap_pred,
                    type_logits,
                    quality_target,
                    overlap_target,
                    type_target=None,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Checkpoint on validation improvement
        if checkpoint_dir is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = checkpoint_dir / "pairwise_scorer_best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                path,
            )
            print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.4f})")

    # Save final model
    if checkpoint_dir is not None:
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            },
            checkpoint_dir / "pairwise_scorer_final.pt",
        )

    return model
