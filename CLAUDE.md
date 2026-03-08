# Robomixer

AI-powered DJ transition detection and mixing engine.

## Project Structure

```
src/robomixer/
    models/         # Pydantic data models (Song, SongAnalysis, TransitionPoint, TransitionScore)
    analysis/       # Audio feature extraction pipeline (beats, key, spectral, stems, VAD)
    scoring/        # Transition scoring (heuristic baseline + ML models)
    training/       # Training data pipeline (YouTube scraping, track ID, dataset curation)
    storage/        # SQLite + Parquet feature store, FAISS index
    api/            # FastAPI service
    cli/            # Typer CLI
```

## Conventions

- Python 3.11+, type hints everywhere, Pydantic v2 models
- Format with `ruff format`, lint with `ruff check`
- Tests in `tests/` mirroring `src/` structure
- Use `polars` not `pandas` for dataframes
- Use `pedalboard` for audio I/O, `librosa` for feature extraction
- All time values in seconds (float), all audio at native sample rate
- Feature store: SQLite for scalars/metadata, Parquet for time-series arrays
- Config via pydantic-settings with env vars prefixed `ROBOMIXER_`

## Key Domain Concepts

- **TransitionPoint**: A beat/phrase-aligned timestamp + context window (energy, spectral, vocals, structure)
- **TransitionScore**: Pairwise score between exit point (song A) and entry point (song B)
- **Mix Embedding**: Per-song vector for fast approximate nearest-neighbor compatibility search
- **Camelot Wheel**: Harmonic mixing system — compatible keys are same code, ±1, or parallel major/minor

## Agent Team Architecture

When spawning an agent team for this project, use these roles:

### audio-analyst
Owns: `analysis/`, `models/song.py`
Focus: Audio feature extraction pipeline — beat/downbeat detection (allin1), key detection (essentia),
spectral features (librosa), source separation (demucs), vocal activity detection (silero-vad),
candidate transition point extraction. Performance-sensitive — profile and optimize extraction times.

### data-engineer
Owns: `training/`, `storage/`
Focus: Training data pipeline — YouTube DJ set scraping (yt-dlp), 1001Tracklists integration,
audio fingerprinting (chromaprint), transition region detection (ruptures), track-to-original alignment.
Also owns the feature store (SQLite schema, Parquet I/O, FAISS index management).

### ml-engineer
Owns: `scoring/embedding.py`, `scoring/pairwise.py`, `training/dataset.py`
Focus: ML model design and training — contrastive learning for mix embeddings, pairwise transition
scorer, PyTorch training loops, evaluation metrics. Works with data-engineer on dataset quality.

### backend-engineer
Owns: `api/`, `cli/`, `config.py`
Focus: FastAPI service, Typer CLI, configuration management, task queue orchestration,
integration between all subsystems. Ensures the pipeline is wired end-to-end.

### scoring-specialist
Owns: `scoring/heuristic.py`, `models/transition.py`
Focus: Transition scoring logic — Camelot harmonic compatibility, BPM matching, energy flow
analysis, vocal clash detection, phrase alignment. Builds the rule-based baseline that works
before ML, and defines the scoring interface that ML models must implement.
