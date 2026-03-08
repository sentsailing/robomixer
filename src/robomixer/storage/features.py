"""Parquet-based storage for time-series audio features."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl

from robomixer.config import settings


class FeatureStore:
    """Read/write time-series features as Parquet files, one per song."""

    def __init__(self, features_dir: Path | None = None) -> None:
        self.features_dir = features_dir or settings.features_dir
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, song_id: UUID) -> Path:
        return self.features_dir / f"{song_id}.parquet"

    def save(self, song_id: UUID, features: dict[str, np.ndarray]) -> None:
        """Save time-series features for a song.

        Args:
            song_id: UUID of the song.
            features: Dict mapping feature name to 1D numpy array.
                      Arrays may be different lengths — padded with NaN.
        """
        max_len = max(len(v) for v in features.values()) if features else 0
        data = {}
        for name, arr in features.items():
            padded = np.full(max_len, np.nan)
            padded[: len(arr)] = arr
            data[name] = padded

        df = pl.DataFrame(data)
        df.write_parquet(self._path(song_id))

    def load(self, song_id: UUID) -> dict[str, np.ndarray]:
        """Load time-series features for a song.

        Returns dict mapping feature name to numpy array (NaN padding stripped).
        """
        path = self._path(song_id)
        if not path.exists():
            return {}

        df = pl.read_parquet(path)
        result = {}
        for col in df.columns:
            arr = df[col].to_numpy()
            # Strip trailing NaN padding
            valid_mask = ~np.isnan(arr)
            if valid_mask.any():
                last_valid = np.where(valid_mask)[0][-1]
                result[col] = arr[: last_valid + 1]
            else:
                result[col] = np.array([])
        return result

    def exists(self, song_id: UUID) -> bool:
        return self._path(song_id).exists()
