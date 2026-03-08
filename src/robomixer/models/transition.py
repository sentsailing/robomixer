"""Transition point and scoring models."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class TransitionType(StrEnum):
    BEATMIX = "beatmix"
    CUT = "cut"
    ECHO_OUT = "echo_out"
    FILTER_SWEEP = "filter_sweep"
    LONG_BLEND = "long_blend"


class EnergyDirection(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


class PointType(StrEnum):
    EXIT = "exit"
    ENTRY = "entry"


class TransitionPoint(BaseModel):
    """A candidate transition point in a song.

    Represents a beat/phrase-aligned timestamp with a context window
    capturing the local audio characteristics around that point.
    """

    song_id: UUID
    timestamp: float  # seconds, the cue point
    point_type: PointType

    # Context window
    window_start: float = 0.0  # seconds
    window_end: float = 0.0  # seconds

    # Local features at this point
    local_bpm: float = 0.0
    local_key: str = ""
    local_energy: float = 0.0
    beats_until_phrase_boundary: int = 0
    current_segment: str = ""  # "outro", "intro", etc.
    has_vocals: bool = False
    spectral_profile: list[float] = Field(default_factory=list)  # MFCC summary

    # Quality indicators
    phrase_aligned: bool = False
    energy_direction: EnergyDirection = EnergyDirection.STABLE
    breakdown_score: float = 0.0  # how "open" / stripped-back the mix is

    def feature_vector(self) -> list[float]:
        """Flatten to a numeric feature vector for ML scoring."""
        return [
            self.local_bpm,
            self.local_energy,
            self.beats_until_phrase_boundary,
            float(self.has_vocals),
            float(self.phrase_aligned),
            {"rising": 1.0, "falling": -1.0, "stable": 0.0}[self.energy_direction],
            self.breakdown_score,
            *self.spectral_profile,
        ]


class TransitionScore(BaseModel):
    """Score for a transition between two songs at specific cue points."""

    exit_song_id: UUID
    entry_song_id: UUID
    exit_timestamp: float
    entry_timestamp: float

    # Component scores (0.0 to 1.0)
    harmonic_score: float = 0.0
    tempo_score: float = 0.0
    energy_score: float = 0.0
    spectral_score: float = 0.0
    structural_score: float = 0.0
    vocal_score: float = 0.0

    # Aggregate
    overall_score: float = 0.0

    # Transition metadata
    suggested_technique: TransitionType = TransitionType.BEATMIX
    overlap_beats: int = 16

    def compute_overall(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted overall score from components."""
        w = weights or {
            "harmonic": 0.25,
            "tempo": 0.20,
            "energy": 0.15,
            "spectral": 0.10,
            "structural": 0.15,
            "vocal": 0.15,
        }
        self.overall_score = (
            w["harmonic"] * self.harmonic_score
            + w["tempo"] * self.tempo_score
            + w["energy"] * self.energy_score
            + w["spectral"] * self.spectral_score
            + w["structural"] * self.structural_score
            + w["vocal"] * self.vocal_score
        )
        return self.overall_score
