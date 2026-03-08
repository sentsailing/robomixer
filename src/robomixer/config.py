"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ROBOMIXER_"}

    # Paths
    library_dir: Path = Path("~/Music/robomixer-library").expanduser()
    data_dir: Path = Path("~/Music/robomixer-data").expanduser()
    db_path: Path = Path("~/Music/robomixer-data/robomixer.db").expanduser()
    features_dir: Path = Path("~/Music/robomixer-data/features").expanduser()

    # Audio analysis
    sample_rate: int = 44100
    demucs_model: str = "htdemucs"
    demucs_device: str = "cpu"  # "cuda" or "cpu"
    skip_separation: bool = False  # skip demucs entirely for fast analysis

    # Transition point extraction
    exit_region_pct: float = 0.75  # candidate exits in last 25% of song
    entry_region_pct: float = 0.40  # candidate entries in first 40% of song
    context_window_beats: int = 16  # beats of context around each cue point

    # Scoring
    max_bpm_diff_pct: float = 0.08  # 8% BPM difference threshold
    vocal_clash_penalty: float = 0.7  # multiplier when both tracks have vocals

    # Training data
    youtube_download_dir: Path = Path("~/Music/robomixer-data/dj-sets").expanduser()
    max_concurrent_downloads: int = 3
    acoustid_api_key: str = ""  # AcoustID API key for fingerprint lookups

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8420

    # Embedding
    embedding_dim: int = 256
    faiss_nprobe: int = 16

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.data_dir, self.features_dir, self.youtube_download_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
