"""Transition scoring and recommendation endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from robomixer.storage.db import Database

router = APIRouter()
db = Database()


class TransitionResponse(BaseModel):
    exit_song_id: str
    entry_song_id: str
    exit_timestamp: float
    entry_timestamp: float
    overall_score: float
    harmonic_score: float
    tempo_score: float
    energy_score: float
    vocal_score: float
    suggested_technique: str
    overlap_beats: int


@router.get("/score")
def score_transition(song_a: UUID, song_b: UUID) -> list[TransitionResponse]:
    """Score all candidate transition points between two songs.

    Returns pre-computed transition scores from the database. Scores are generated
    during import/analysis and cached. Returns empty list if no scores exist yet.
    """
    scores = db.get_transition_scores(song_a, song_b)
    if not scores:
        # Check that both songs exist
        if db.get_song(song_a) is None:
            raise HTTPException(status_code=404, detail=f"Song {song_a} not found")
        if db.get_song(song_b) is None:
            raise HTTPException(status_code=404, detail=f"Song {song_b} not found")
    return [
        TransitionResponse(
            exit_song_id=str(s.exit_song_id),
            entry_song_id=str(s.entry_song_id),
            exit_timestamp=s.exit_timestamp,
            entry_timestamp=s.entry_timestamp,
            overall_score=s.overall_score,
            harmonic_score=s.harmonic_score,
            tempo_score=s.tempo_score,
            energy_score=s.energy_score,
            vocal_score=s.vocal_score,
            suggested_technique=s.suggested_technique,
            overlap_beats=s.overlap_beats,
        )
        for s in scores
    ]


@router.get("/best")
def best_transitions(song_id: UUID, limit: int = 10) -> list[TransitionResponse]:
    """Find the best transitions from a given song to any other song in the library.

    Returns the top-scoring pre-computed transitions, sorted by overall_score descending.
    """
    if db.get_song(song_id) is None:
        raise HTTPException(status_code=404, detail=f"Song {song_id} not found")

    scores = db.get_best_transitions(song_id, limit=limit)
    return [
        TransitionResponse(
            exit_song_id=str(s.exit_song_id),
            entry_song_id=str(s.entry_song_id),
            exit_timestamp=s.exit_timestamp,
            entry_timestamp=s.entry_timestamp,
            overall_score=s.overall_score,
            harmonic_score=s.harmonic_score,
            tempo_score=s.tempo_score,
            energy_score=s.energy_score,
            vocal_score=s.vocal_score,
            suggested_technique=s.suggested_technique,
            overlap_beats=s.overlap_beats,
        )
        for s in scores
    ]
