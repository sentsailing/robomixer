"""Mix embedding model for fast approximate compatibility search.

Maps each song to a dense vector such that songs that mix well together
are close in embedding space. Trained with contrastive loss on real DJ
set track sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from robomixer.config import settings


class MixEmbeddingNet(nn.Module):
    """Projects song features to a mix-compatibility embedding space."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        embed_dim = settings.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed song features. Output is L2-normalized."""
        emb = self.net(x)
        return nn.functional.normalize(emb, p=2, dim=-1)


class PairwiseScorer(nn.Module):
    """Scores a (exit_point, entry_point) feature pair for transition quality."""

    def __init__(self, point_feature_dim: int = 40) -> None:
        super().__init__()
        combined_dim = point_feature_dim * 2
        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.quality_head = nn.Linear(64, 1)  # transition quality score
        self.overlap_head = nn.Linear(64, 1)  # suggested overlap beats
        self.type_head = nn.Linear(64, 5)  # transition type classification

    def forward(
        self, exit_features: torch.Tensor, entry_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score a transition.

        Returns (quality_score, overlap_beats, transition_type_logits).
        """
        combined = torch.cat([exit_features, entry_features], dim=-1)
        hidden = self.scorer(combined)
        quality = torch.sigmoid(self.quality_head(hidden))
        overlap = torch.relu(self.overlap_head(hidden)) + 4  # minimum 4 beats
        transition_type = self.type_head(hidden)
        return quality, overlap, transition_type
