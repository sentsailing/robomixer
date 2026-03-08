"""Contrastive training loop for MixEmbeddingNet.

Uses InfoNCE (NT-Xent) loss on (anchor, positive) pairs from real DJ sets.
Negatives are drawn from other examples in the same batch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as f
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from robomixer.scoring.embedding import MixEmbeddingNet
from robomixer.training.dataset import ContrastivePairDataset


def info_nce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE / NT-Xent loss.

    Uses in-batch negatives: for each anchor, the positive is its paired
    embedding and all other positives in the batch serve as negatives.

    Args:
        anchor_emb: (B, D) L2-normalized anchor embeddings.
        positive_emb: (B, D) L2-normalized positive embeddings.
        temperature: Softmax temperature scaling.

    Returns:
        Scalar loss.
    """
    # Similarity matrix: (B, B)
    logits = anchor_emb @ positive_emb.T / temperature
    # Labels: diagonal entries are the positives
    labels = torch.arange(logits.size(0), device=logits.device)
    return f.cross_entropy(logits, labels)


def train_embedding(
    dataset: ContrastivePairDataset,
    *,
    input_dim: int = 64,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.07,
    val_fraction: float = 0.1,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
) -> MixEmbeddingNet:
    """Train MixEmbeddingNet with InfoNCE contrastive loss.

    Args:
        dataset: ContrastivePairDataset of positive pairs.
        input_dim: Song feature vector dimension.
        hidden_dim: Hidden layer dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size (larger = more negatives).
        lr: Initial learning rate for Adam.
        temperature: InfoNCE temperature parameter.
        val_fraction: Fraction of data for validation.
        checkpoint_dir: Directory to save model checkpoints.
        device: torch device string. Auto-selects CUDA if available.

    Returns:
        Trained MixEmbeddingNet.
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

    model = MixEmbeddingNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)

            loss = info_nce_loss(anchor_emb, positive_emb, temperature)

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
                anchor = batch["anchor"].to(device)
                positive = batch["positive"].to(device)

                anchor_emb = model(anchor)
                positive_emb = model(positive)

                loss = info_nce_loss(anchor_emb, positive_emb, temperature)
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
            path = checkpoint_dir / "mix_embedding_best.pt"
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
            checkpoint_dir / "mix_embedding_final.pt",
        )

    return model
