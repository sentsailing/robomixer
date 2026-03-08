"""Training data models for DJ set analysis."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DJSet(BaseModel):
    """A scraped DJ set from YouTube."""

    set_id: UUID = Field(default_factory=uuid4)
    source_url: str
    dj_name: str = ""
    title: str = ""
    genre: str = ""
    duration_sec: float = 0.0
    download_date: datetime = Field(default_factory=datetime.now)
    audio_path: str = ""
    tracklist_source: str = ""  # "1001tracklists", "youtube_description", "fingerprint"


class DJSetTransition(BaseModel):
    """A labeled transition extracted from a DJ set, used for training."""

    set_id: UUID
    transition_id: UUID = Field(default_factory=uuid4)

    # Track identification
    track_a_id: str = ""  # MusicBrainz ID or fingerprint hash
    track_a_title: str = ""
    track_b_id: str = ""
    track_b_title: str = ""

    # Position in the mix
    transition_start: float = 0.0  # seconds in the mix
    transition_end: float = 0.0
    transition_duration: float = 0.0

    # Mapped to individual tracks
    exit_time_in_a: float = 0.0  # seconds into track A
    entry_time_in_b: float = 0.0  # seconds into track B

    # Quality proxy
    quality_proxy: float = 0.5  # 0-1, derived from DJ reputation / set engagement
    confidence: float = 0.0  # how confident we are in the label accuracy
