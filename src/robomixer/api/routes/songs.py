"""Song import and analysis endpoints."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from robomixer.analysis.pipeline import SUPPORTED_EXTENSIONS
from robomixer.cli.metadata import read_metadata
from robomixer.models.song import Song
from robomixer.storage.db import Database

router = APIRouter()
db = Database()


class SongImportRequest(BaseModel):
    file_path: str


class SongResponse(BaseModel):
    song_id: str
    title: str
    artist: str
    duration_sec: float
    bpm: float | None = None
    key: str | None = None
    camelot_code: str | None = None


def _song_to_response(song: Song) -> SongResponse:
    """Convert a Song to a SongResponse, enriching with analysis data if available."""
    analysis = db.get_analysis(song.song_id)
    return SongResponse(
        song_id=str(song.song_id),
        title=song.title,
        artist=song.artist,
        duration_sec=song.duration_sec,
        bpm=analysis.bpm if analysis and analysis.bpm else None,
        key=analysis.key if analysis and analysis.key else None,
        camelot_code=analysis.camelot_code if analysis and analysis.camelot_code else None,
    )


@router.get("/")
def list_songs() -> list[SongResponse]:
    songs = db.list_songs()
    return [_song_to_response(s) for s in songs]


@router.post("/", status_code=201)
def import_song(request: SongImportRequest) -> SongResponse:
    """Import a song from a local file path. Reads audio metadata tags."""
    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    meta = read_metadata(path)
    song = Song(
        file_path=path.resolve(),
        title=meta.title,
        artist=meta.artist,
        duration_sec=meta.duration_sec,
        sample_rate=meta.sample_rate,
    )
    db.insert_song(song)
    return _song_to_response(song)


@router.get("/{song_id}")
def get_song(song_id: UUID) -> SongResponse:
    song = db.get_song(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")
    return _song_to_response(song)
