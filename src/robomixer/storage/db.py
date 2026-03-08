"""SQLite database schema and queries for song metadata and analysis scalars."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from robomixer.config import settings
from robomixer.models.song import Song, SongAnalysis, SongSegment, VocalRegion
from robomixer.models.transition import TransitionScore, TransitionType

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    song_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    title TEXT DEFAULT '',
    artist TEXT DEFAULT '',
    duration_sec REAL DEFAULT 0,
    sample_rate INTEGER DEFAULT 44100,
    import_date TEXT NOT NULL,
    analysis_version INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS song_analysis (
    song_id TEXT PRIMARY KEY REFERENCES songs(song_id),
    bpm REAL DEFAULT 0,
    bpm_confidence REAL DEFAULT 0,
    key TEXT DEFAULT '',
    key_confidence REAL DEFAULT 0,
    camelot_code TEXT DEFAULT '',
    average_energy REAL DEFAULT 0,
    segments_json TEXT DEFAULT '[]',
    vocal_regions_json TEXT DEFAULT '[]',
    mix_embedding BLOB
);

CREATE TABLE IF NOT EXISTS transition_scores (
    exit_song_id TEXT REFERENCES songs(song_id),
    entry_song_id TEXT REFERENCES songs(song_id),
    exit_timestamp REAL,
    entry_timestamp REAL,
    overall_score REAL,
    harmonic_score REAL,
    tempo_score REAL,
    energy_score REAL,
    spectral_score REAL,
    structural_score REAL,
    vocal_score REAL,
    suggested_technique TEXT DEFAULT 'beatmix',
    overlap_beats INTEGER DEFAULT 16,
    PRIMARY KEY (exit_song_id, entry_song_id, exit_timestamp, entry_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_transition_overall
    ON transition_scores(exit_song_id, overall_score DESC);

CREATE TABLE IF NOT EXISTS dj_sets (
    set_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    dj_name TEXT DEFAULT '',
    title TEXT DEFAULT '',
    genre TEXT DEFAULT '',
    duration_sec REAL DEFAULT 0,
    download_date TEXT NOT NULL,
    audio_path TEXT DEFAULT '',
    tracklist_source TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dj_set_transitions (
    transition_id TEXT PRIMARY KEY,
    set_id TEXT REFERENCES dj_sets(set_id),
    track_a_id TEXT DEFAULT '',
    track_a_title TEXT DEFAULT '',
    track_b_id TEXT DEFAULT '',
    track_b_title TEXT DEFAULT '',
    transition_start REAL DEFAULT 0,
    transition_end REAL DEFAULT 0,
    transition_duration REAL DEFAULT 0,
    exit_time_in_a REAL DEFAULT 0,
    entry_time_in_b REAL DEFAULT 0,
    quality_proxy REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0
);
"""


def _embedding_to_blob(embedding: list[float]) -> bytes | None:
    """Pack a float list into a binary blob (little-endian float32)."""
    if not embedding:
        return None
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes | None) -> list[float]:
    """Unpack a binary blob into a float list."""
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class Database:
    """SQLite database for robomixer metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_song(self, song: Song) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO songs
                   (song_id, file_path, title, artist, duration_sec,
                    sample_rate, import_date, analysis_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(song.song_id),
                    str(song.file_path),
                    song.title,
                    song.artist,
                    song.duration_sec,
                    song.sample_rate,
                    song.import_date.isoformat(),
                    song.analysis_version,
                ),
            )

    def get_song(self, song_id: UUID) -> Song | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM songs WHERE song_id = ?", (str(song_id),)).fetchone()
            if row is None:
                return None
            return Song(
                song_id=UUID(row["song_id"]),
                file_path=Path(row["file_path"]),
                title=row["title"],
                artist=row["artist"],
                duration_sec=row["duration_sec"],
                sample_rate=row["sample_rate"],
                analysis_version=row["analysis_version"],
            )

    def list_songs(self) -> list[Song]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM songs ORDER BY import_date DESC").fetchall()
            return [
                Song(
                    song_id=UUID(row["song_id"]),
                    file_path=Path(row["file_path"]),
                    title=row["title"],
                    artist=row["artist"],
                    duration_sec=row["duration_sec"],
                    sample_rate=row["sample_rate"],
                    analysis_version=row["analysis_version"],
                )
                for row in rows
            ]

    def insert_analysis(self, analysis: SongAnalysis) -> None:
        """Insert or replace a song analysis record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO song_analysis
                   (song_id, bpm, bpm_confidence, key, key_confidence, camelot_code,
                    average_energy, segments_json, vocal_regions_json, mix_embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(analysis.song_id),
                    analysis.bpm,
                    analysis.bpm_confidence,
                    analysis.key,
                    analysis.key_confidence,
                    analysis.camelot_code,
                    analysis.average_energy,
                    json.dumps([s.model_dump() for s in analysis.segments]),
                    json.dumps([v.model_dump() for v in analysis.vocal_regions]),
                    _embedding_to_blob(analysis.mix_embedding),
                ),
            )

    def get_analysis(self, song_id: UUID) -> SongAnalysis | None:
        """Retrieve a song analysis by song ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM song_analysis WHERE song_id = ?", (str(song_id),)
            ).fetchone()
            if row is None:
                return None
            return SongAnalysis(
                song_id=UUID(row["song_id"]),
                bpm=row["bpm"],
                bpm_confidence=row["bpm_confidence"],
                key=row["key"],
                key_confidence=row["key_confidence"],
                camelot_code=row["camelot_code"],
                average_energy=row["average_energy"],
                segments=[SongSegment(**s) for s in json.loads(row["segments_json"] or "[]")],
                vocal_regions=[
                    VocalRegion(**v) for v in json.loads(row["vocal_regions_json"] or "[]")
                ],
                mix_embedding=_blob_to_embedding(row["mix_embedding"]),
            )

    def list_analyses(self) -> list[SongAnalysis]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM song_analysis").fetchall()
            return [
                SongAnalysis(
                    song_id=UUID(row["song_id"]),
                    bpm=row["bpm"],
                    bpm_confidence=row["bpm_confidence"],
                    key=row["key"],
                    key_confidence=row["key_confidence"],
                    camelot_code=row["camelot_code"],
                    average_energy=row["average_energy"],
                    segments=[SongSegment(**s) for s in json.loads(row["segments_json"] or "[]")],
                    vocal_regions=[
                        VocalRegion(**v) for v in json.loads(row["vocal_regions_json"] or "[]")
                    ],
                    mix_embedding=_blob_to_embedding(row["mix_embedding"]),
                )
                for row in rows
            ]

    def insert_transition_score(self, score: TransitionScore) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO transition_scores
                   (exit_song_id, entry_song_id, exit_timestamp, entry_timestamp,
                    overall_score, harmonic_score, tempo_score, energy_score,
                    spectral_score, structural_score, vocal_score,
                    suggested_technique, overlap_beats)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(score.exit_song_id),
                    str(score.entry_song_id),
                    score.exit_timestamp,
                    score.entry_timestamp,
                    score.overall_score,
                    score.harmonic_score,
                    score.tempo_score,
                    score.energy_score,
                    score.spectral_score,
                    score.structural_score,
                    score.vocal_score,
                    score.suggested_technique,
                    score.overlap_beats,
                ),
            )

    def get_transition_scores(self, song_a: UUID, song_b: UUID) -> list[TransitionScore]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transition_scores
                   WHERE exit_song_id = ? AND entry_song_id = ?
                   ORDER BY overall_score DESC""",
                (str(song_a), str(song_b)),
            ).fetchall()
            return [self._row_to_transition_score(row) for row in rows]

    def get_top_transitions(self, song_id: UUID, limit: int = 10) -> list[TransitionScore]:
        """Get top-scored transitions from a given song, ordered by overall_score."""
        return self.get_best_transitions(song_id, limit)

    def get_best_transitions(self, song_id: UUID, limit: int = 10) -> list[TransitionScore]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transition_scores
                   WHERE exit_song_id = ?
                   ORDER BY overall_score DESC
                   LIMIT ?""",
                (str(song_id), limit),
            ).fetchall()
            return [self._row_to_transition_score(row) for row in rows]

    @staticmethod
    def _row_to_transition_score(row: sqlite3.Row) -> TransitionScore:
        return TransitionScore(
            exit_song_id=UUID(row["exit_song_id"]),
            entry_song_id=UUID(row["entry_song_id"]),
            exit_timestamp=row["exit_timestamp"],
            entry_timestamp=row["entry_timestamp"],
            overall_score=row["overall_score"],
            harmonic_score=row["harmonic_score"],
            tempo_score=row["tempo_score"],
            energy_score=row["energy_score"],
            spectral_score=row["spectral_score"],
            structural_score=row["structural_score"],
            vocal_score=row["vocal_score"],
            suggested_technique=TransitionType(row["suggested_technique"]),
            overlap_beats=row["overlap_beats"],
        )
