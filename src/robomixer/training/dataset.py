"""PyTorch Dataset classes for training ML models."""

from __future__ import annotations

from uuid import UUID

import numpy as np
import torch
from torch.utils.data import Dataset

from robomixer.models.training import DJSetTransition
from robomixer.storage.features import FeatureStore

# Default feature dimensions — must match model architectures
SONG_FEATURE_DIM = 64  # MixEmbeddingNet input_dim
POINT_FEATURE_DIM = 40  # PairwiseScorer point_feature_dim


def _load_song_features(
    feature_store: FeatureStore, track_id: str, dim: int = SONG_FEATURE_DIM
) -> torch.Tensor:
    """Load and pad/truncate a song-level feature vector from the feature store.

    The feature store holds per-song time-series features in Parquet. For the
    embedding model we need a fixed-size summary vector. We concatenate means of
    all available feature columns and pad/truncate to ``dim``.
    """
    try:
        features = feature_store.load(UUID(track_id))
    except (ValueError, KeyError):
        return torch.zeros(dim)

    if not features:
        return torch.zeros(dim)

    # Build summary vector: mean of each feature column
    parts = [np.nanmean(arr) for arr in features.values() if len(arr) > 0]
    vec = np.array(parts, dtype=np.float32)

    # Pad or truncate to fixed dim
    if len(vec) >= dim:
        vec = vec[:dim]
    else:
        vec = np.pad(vec, (0, dim - len(vec)), constant_values=0.0)

    return torch.from_numpy(vec)


def _load_transition_features(
    feature_store: FeatureStore,
    track_id: str,
    time_in_track: float,
    dim: int = POINT_FEATURE_DIM,
) -> torch.Tensor:
    """Load a transition point feature vector from the feature store.

    Extracts features around ``time_in_track`` and constructs a fixed-size
    vector comparable to TransitionPoint.feature_vector().
    """
    try:
        features = feature_store.load(UUID(track_id))
    except (ValueError, KeyError):
        return torch.zeros(dim)

    if not features:
        return torch.zeros(dim)

    # Sample feature values at the given timestamp
    # Assume features contain arrays indexed by time; we pick the nearest sample
    parts: list[float] = []
    for arr in features.values():
        if len(arr) == 0:
            parts.append(0.0)
            continue
        # Estimate sample index from time (assume ~1 value per analysis hop)
        idx = min(int(time_in_track), len(arr) - 1)
        idx = max(0, idx)
        parts.append(float(arr[idx]))

    vec = np.array(parts, dtype=np.float32)

    if len(vec) >= dim:
        vec = vec[:dim]
    else:
        vec = np.pad(vec, (0, dim - len(vec)), constant_values=0.0)

    return torch.from_numpy(vec)


class TransitionDataset(Dataset):
    """Dataset of labeled DJ transitions for training the pairwise scorer."""

    def __init__(
        self,
        transitions: list[DJSetTransition],
        feature_store: FeatureStore | None = None,
    ) -> None:
        self.transitions = transitions
        self.feature_store = feature_store or FeatureStore()

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = self.transitions[idx]

        exit_features = _load_transition_features(
            self.feature_store, t.track_a_id, t.exit_time_in_a
        )
        entry_features = _load_transition_features(
            self.feature_store, t.track_b_id, t.entry_time_in_b
        )

        return {
            "exit_features": exit_features,
            "entry_features": entry_features,
            "quality": torch.tensor(t.quality_proxy, dtype=torch.float32),
            "overlap_beats": torch.tensor(t.transition_duration, dtype=torch.float32),
        }


class ContrastivePairDataset(Dataset):
    """Dataset of (track_A, track_B) pairs for training the mix embedding model.

    Positive pairs: consecutive tracks in real DJ sets.
    Negative pairs: random pairs that never appear together.
    """

    def __init__(
        self,
        positive_pairs: list[tuple[str, str]],
        all_track_ids: list[str],
        feature_store: FeatureStore | None = None,
    ) -> None:
        self.positive_pairs = positive_pairs
        self.all_track_ids = all_track_ids
        self.feature_store = feature_store or FeatureStore()

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track_a_id, track_b_id = self.positive_pairs[idx]

        anchor = _load_song_features(self.feature_store, track_a_id)
        positive = _load_song_features(self.feature_store, track_b_id)

        return {
            "anchor": anchor,
            "positive": positive,
        }
