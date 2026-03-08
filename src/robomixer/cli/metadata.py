"""Audio file metadata extraction using tinytag.

Reads ID3/Vorbis/MP4 tags to populate Song title, artist, and duration
during import, without loading the full audio waveform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    title: str
    artist: str
    album: str
    duration_sec: float
    sample_rate: int
    channels: int
    bitrate: float  # kbps


def read_metadata(path: Path) -> AudioMetadata:
    """Extract metadata tags and technical info from an audio file.

    Falls back to filename-based title if tags are missing.
    """
    from tinytag import TinyTag

    tag = TinyTag.get(str(path))

    title = (tag.title or "").strip()
    artist = (tag.artist or "").strip()

    # Fall back to filename if no title tag
    if not title:
        title = path.stem
        # Try to parse "Artist - Title" from filename
        if " - " in title and not artist:
            parts = title.split(" - ", 1)
            artist = parts[0].strip()
            title = parts[1].strip()

    return AudioMetadata(
        title=title,
        artist=artist,
        album=(tag.album or "").strip(),
        duration_sec=tag.duration or 0.0,
        sample_rate=tag.samplerate or 44100,
        channels=tag.channels or 2,
        bitrate=tag.bitrate or 0.0,
    )
