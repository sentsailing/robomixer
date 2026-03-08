"""Song and analysis data models."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field


class SegmentLabel(StrEnum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    BREAK = "break"
    INSTRUMENTAL = "inst"
    SOLO = "solo"


class SongSegment(BaseModel):
    """A structural segment within a song."""

    label: SegmentLabel
    start: float  # seconds
    end: float  # seconds


class VocalRegion(BaseModel):
    """A region where vocals are detected."""

    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0


class Song(BaseModel):
    """Core song record."""

    song_id: UUID = Field(default_factory=uuid4)
    file_path: Path
    title: str = ""
    artist: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 44100
    import_date: datetime = Field(default_factory=datetime.now)
    analysis_version: int = 1

    @staticmethod
    def content_hash(file_path: Path) -> str:
        """Generate a content hash for deduplication."""
        h = hashlib.blake2b(digest_size=16)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class SongAnalysis(BaseModel):
    """Full analysis results for a song. Time-series arrays stored separately as Parquet."""

    model_config = {"arbitrary_types_allowed": True}

    song_id: UUID

    # Global features
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    key: str = ""  # e.g. "Cm", "G"
    key_confidence: float = 0.0
    camelot_code: str = ""  # e.g. "5A"
    average_energy: float = 0.0

    # Structural segments
    segments: list[SongSegment] = Field(default_factory=list)

    # Vocal regions
    vocal_regions: list[VocalRegion] = Field(default_factory=list)

    # Beat grid (seconds) — stored in Parquet for large arrays, kept here for small ones
    beat_times: list[float] = Field(default_factory=list)
    downbeat_times: list[float] = Field(default_factory=list)

    # Mix embedding (populated after ML training)
    mix_embedding: list[float] = Field(default_factory=list)

    def has_vocals_at(self, timestamp: float) -> bool:
        """Check if vocals are active at a given timestamp."""
        return any(r.start <= timestamp <= r.end for r in self.vocal_regions)

    def segment_at(self, timestamp: float) -> SongSegment | None:
        """Get the structural segment at a given timestamp."""
        for seg in self.segments:
            if seg.start <= timestamp <= seg.end:
                return seg
        return None

    def nearest_downbeat(self, timestamp: float) -> float:
        """Find the nearest downbeat to a timestamp."""
        if not self.downbeat_times:
            return timestamp
        arr = np.array(self.downbeat_times)
        idx = int(np.argmin(np.abs(arr - timestamp)))
        return self.downbeat_times[idx]
