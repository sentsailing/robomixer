"""Orchestrator for the per-song audio analysis pipeline.

Sequences all feature extraction steps:
1. Load audio
2. Source separation (demucs) → stems
3. Beat/downbeat/structure analysis (allin1)
4. Key detection (essentia)
5. Spectral features (librosa)
6. Vocal activity detection (silero-vad on vocal stem)
7. Candidate transition point extraction
8. Store results to feature store
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from robomixer.config import settings
from robomixer.models.song import Song, SongAnalysis

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus", ".wma"}

# Maps (key_name, scale) from essentia to Camelot wheel codes.
# Essentia returns key as e.g. "C" with scale "minor" or "major".
KEY_TO_CAMELOT: dict[tuple[str, str], str] = {
    ("Ab", "major"): "4B",
    ("B", "major"): "1B",
    ("Eb", "major"): "3B",
    ("Bb", "major"): "6B",
    ("F", "major"): "7B",
    ("C", "major"): "8B",
    ("G", "major"): "9B",
    ("D", "major"): "10B",
    ("A", "major"): "11B",
    ("E", "major"): "12B",
    ("Gb", "major"): "2B",
    ("Db", "major"): "5B",
    ("Ab", "minor"): "1A",
    ("Eb", "minor"): "2A",
    ("Bb", "minor"): "3A",
    ("F", "minor"): "4A",
    ("C", "minor"): "5A",
    ("G", "minor"): "6A",
    ("D", "minor"): "7A",
    ("A", "minor"): "8A",
    ("E", "minor"): "9A",
    ("B", "minor"): "10A",
    ("Gb", "minor"): "11A",
    ("Db", "minor"): "12A",
    # Enharmonic equivalents
    ("G#", "major"): "4B",
    ("F#", "major"): "2B",
    ("C#", "major"): "5B",
    ("G#", "minor"): "1A",
    ("D#", "minor"): "2A",
    ("A#", "minor"): "3A",
    ("F#", "minor"): "11A",
    ("C#", "minor"): "12A",
}

# Map allin1 segment labels to our SegmentLabel enum values
_ALLIN1_LABEL_MAP: dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "outro": "outro",
    "break": "break",
    "inst": "inst",
    "instrumental": "inst",
    "solo": "solo",
}


class AnalysisPipeline:
    """Runs the full analysis pipeline for a song."""

    def __init__(self) -> None:
        self._initialized = False
        self._demucs_model = None

    def _lazy_init(self) -> None:
        """Defer heavy imports until first use."""
        if self._initialized:
            return
        self._load_demucs_model()
        self._initialized = True

    def _load_demucs_model(self) -> None:
        """Load the demucs separation model."""
        import torch
        from demucs.pretrained import get_model

        device = settings.demucs_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"

        self._demucs_model = get_model(settings.demucs_model)
        self._demucs_device = device
        self._demucs_model.to(device)

    def analyze(self, song: Song) -> SongAnalysis:
        """Run full analysis pipeline on a song.

        Returns a SongAnalysis with all extracted features populated.
        """
        self._lazy_init()

        analysis = SongAnalysis(song_id=song.song_id)

        # Step 1: Load audio
        audio, sr = self._load_audio(song.file_path)

        # Step 2: Source separation
        stems = self._separate_stems(audio, sr)

        # Step 3: Beat/structure analysis (allin1 needs the file path)
        analysis = self._analyze_structure(song.file_path, analysis)

        # Step 4: Key detection
        analysis = self._detect_key(audio, sr, analysis)

        # Step 5: Spectral features
        analysis = self._extract_spectral(audio, sr, analysis)

        # Step 6: Vocal activity detection
        analysis = self._detect_vocals(stems.get("vocals"), sr, analysis)

        return analysis

    def _load_audio(self, path: Path) -> tuple[NDArray[np.float32], int]:
        """Load audio file, return (audio_array, sample_rate).

        Uses pedalboard for I/O. Returns mono audio as a 1-D float32 numpy array
        at the native sample rate.
        """
        from pedalboard.io import AudioFile

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        with AudioFile(str(path)) as f:
            sr = int(f.samplerate)
            audio = f.read(f.frames)  # shape: (channels, samples)

        # Convert to mono by averaging channels
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)
        audio = audio.squeeze(0)  # (samples,)

        logger.info("Loaded %s: %.1fs @ %d Hz", path.name, len(audio) / sr, sr)
        return audio.astype(np.float32), sr

    def _separate_stems(
        self, audio: NDArray[np.float32], sr: int
    ) -> dict[str, NDArray[np.float32]]:
        """Run demucs source separation. Returns dict of stem name -> audio array.

        Stem names: vocals, drums, bass, other
        Each stem is a 1-D float32 numpy array (mono).
        """
        import torch
        from demucs.apply import apply_model

        model = self._demucs_model

        # Demucs expects (batch, channels, samples) at the model's sample rate
        model_sr = model.samplerate

        # Resample if needed
        if sr != model_sr:
            import librosa

            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
        else:
            audio_resampled = audio

        # Convert mono to stereo (demucs expects 2-channel input)
        stereo = np.stack([audio_resampled, audio_resampled], axis=0)  # (2, samples)
        tensor = torch.from_numpy(stereo).unsqueeze(0).to(self._demucs_device)  # (1, 2, samples)

        with torch.no_grad():
            estimates = apply_model(model, tensor, device=self._demucs_device)

        # estimates shape: (1, n_sources, 2, samples)
        source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        stems: dict[str, NDArray[np.float32]] = {}
        for i, name in enumerate(source_names):
            stem_stereo = estimates[0, i].cpu().numpy()  # (2, samples)
            stem_mono = np.mean(stem_stereo, axis=0).astype(np.float32)  # (samples,)

            # Resample back to original sample rate if we resampled earlier
            if sr != model_sr:
                stem_mono = librosa.resample(stem_mono, orig_sr=model_sr, target_sr=sr).astype(
                    np.float32
                )

            stems[name] = stem_mono

        logger.info("Source separation complete: %s", list(stems.keys()))
        return stems

    def _analyze_structure(self, path: Path, analysis: SongAnalysis) -> SongAnalysis:
        """Extract beats, downbeats, and structural segments via allin1."""
        import allin1

        result = allin1.analyze(path)

        analysis.beat_times = [float(t) for t in result.beats]
        analysis.downbeat_times = [float(t) for t in result.downbeats]
        analysis.bpm = float(result.bpm)

        from robomixer.models.song import SegmentLabel, SongSegment

        for seg in result.segments:
            label_str = _ALLIN1_LABEL_MAP.get(seg.label.lower())
            if label_str is None:
                continue
            analysis.segments.append(
                SongSegment(
                    label=SegmentLabel(label_str),
                    start=float(seg.start),
                    end=float(seg.end),
                )
            )

        logger.info(
            "Structure: %.1f BPM, %d beats, %d downbeats, %d segments",
            analysis.bpm,
            len(analysis.beat_times),
            len(analysis.downbeat_times),
            len(analysis.segments),
        )
        return analysis

    def _detect_key(
        self, audio: NDArray[np.float32], sr: int, analysis: SongAnalysis
    ) -> SongAnalysis:
        """Detect musical key and map to Camelot code via essentia."""
        import essentia.standard as es

        # Essentia expects mono float32, which we already have
        key_extractor = es.KeyExtractor(sampleRate=sr)
        key, scale, strength = key_extractor(audio)

        if scale == "minor":
            analysis.key = f"{key}m"
        else:
            analysis.key = key
        analysis.key_confidence = float(strength)
        analysis.camelot_code = KEY_TO_CAMELOT.get((key, scale), "")

        logger.info(
            "Key: %s (confidence %.2f) -> Camelot %s",
            analysis.key,
            analysis.key_confidence,
            analysis.camelot_code,
        )
        return analysis

    def _extract_spectral(
        self, audio: NDArray[np.float32], sr: int, analysis: SongAnalysis
    ) -> SongAnalysis:
        """Extract spectral features (RMS energy, MFCC, chroma, centroid) via librosa."""
        import librosa

        rms = librosa.feature.rms(y=audio)
        analysis.average_energy = float(np.mean(rms))

        logger.info("Spectral: avg energy %.4f", analysis.average_energy)
        return analysis

    def _detect_vocals(
        self,
        vocal_stem: NDArray[np.float32] | None,
        sr: int,
        analysis: SongAnalysis,
    ) -> SongAnalysis:
        """Detect vocal activity regions using silero-vad on the isolated vocal stem."""
        if vocal_stem is None:
            return analysis

        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()

        # silero-vad requires 16kHz mono audio
        vad_sr = 16000
        if sr != vad_sr:
            import librosa

            vocal_16k = librosa.resample(vocal_stem, orig_sr=sr, target_sr=vad_sr).astype(
                np.float32
            )
        else:
            vocal_16k = vocal_stem

        wav_tensor = torch.from_numpy(vocal_16k)
        timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=vad_sr)

        from robomixer.models.song import VocalRegion

        for ts in timestamps:
            analysis.vocal_regions.append(
                VocalRegion(
                    start=float(ts["start"]) / vad_sr,
                    end=float(ts["end"]) / vad_sr,
                )
            )

        logger.info("VAD: %d vocal regions detected", len(analysis.vocal_regions))
        return analysis
