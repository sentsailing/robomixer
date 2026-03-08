"""Cloud GPU analysis via Modal.

Offloads the heavy pipeline steps (demucs source separation, silero-vad)
to a cloud GPU, then downloads results locally. Local steps (librosa,
essentia) still run locally since they're fast on CPU.

Usage:
    pip install modal
    modal setup          # one-time auth
    robomixer import-song song.mp3 --cloud
"""

from __future__ import annotations

import io
import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_modal():
    """Check that modal is installed and configured."""
    try:
        import modal  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Modal is not installed. Run: pip install modal && modal setup"
        )


def run_cloud_separation(audio_path: Path) -> dict[str, np.ndarray]:
    """Upload audio to Modal, run demucs on GPU, return stems as numpy arrays.

    Returns dict mapping stem name -> 1D float32 numpy array (mono).
    """
    _ensure_modal()
    import modal

    app = modal.App.lookup("robomixer-gpu", create_if_missing=False)
    separate_fn = modal.Function.from_name("robomixer-gpu", "separate_and_detect")

    # Read audio file as bytes
    audio_bytes = audio_path.read_bytes()
    suffix = audio_path.suffix

    # Call remote GPU function
    logger.info("Uploading %s to Modal for GPU processing...", audio_path.name)
    result = separate_fn.remote(audio_bytes, suffix)

    # Decode stems from packed float32 bytes
    stems: dict[str, np.ndarray] = {}
    for name, data in result["stems"].items():
        arr = np.frombuffer(data, dtype=np.float32)
        stems[name] = arr

    # Decode vocal regions
    vocal_regions = result["vocal_regions"]

    logger.info(
        "Cloud processing complete: %d stems, %d vocal regions",
        len(stems),
        len(vocal_regions),
    )
    return stems, vocal_regions


# ---------------------------------------------------------------------------
# Modal app definition — this is what runs in the cloud
# ---------------------------------------------------------------------------

MODAL_APP_CODE = '''
import modal
import io
import struct
import numpy as np

app = modal.App("robomixer-gpu")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0",
        "torchaudio>=2.0",
        "demucs>=4.0",
        "silero-vad>=5.1",
        "librosa>=0.10",
        "soundfile>=0.12",
        "numpy>=1.24",
    )
)


@app.function(image=image, gpu="T4", timeout=300, memory=8192)
def separate_and_detect(audio_bytes: bytes, suffix: str) -> dict:
    """Run demucs source separation + silero VAD on GPU.

    Takes raw audio file bytes, returns stems as packed float32 bytes
    and vocal regions as a list of dicts.
    """
    import tempfile
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from silero_vad import load_silero_vad, get_speech_timestamps

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    # Load audio
    waveform, sr = torchaudio.load(tmp_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # --- Demucs source separation ---
    model = get_model("htdemucs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_sr = model.samplerate
    if sr != model_sr:
        waveform_resampled = torchaudio.functional.resample(waveform, sr, model_sr)
    else:
        waveform_resampled = waveform

    # Demucs expects stereo
    stereo = waveform_resampled.expand(2, -1).unsqueeze(0).to(device)

    with torch.inference_mode():
        estimates = apply_model(model, stereo, device=device)

    # Pack stems as bytes (mono, float32)
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
    stems = {}
    vocal_stem_mono = None
    for i, name in enumerate(source_names):
        stem = estimates[0, i].cpu()
        mono = stem.mean(dim=0).numpy().astype(np.float32)

        # Resample back if needed
        if sr != model_sr:
            import librosa
            mono = librosa.resample(mono, orig_sr=model_sr, target_sr=sr).astype(np.float32)

        stems[name] = mono.tobytes()
        if name == "vocals":
            vocal_stem_mono = mono

    # --- Silero VAD on vocal stem ---
    vocal_regions = []
    if vocal_stem_mono is not None:
        vad_model = load_silero_vad()
        vad_sr = 16000
        if sr != vad_sr:
            import librosa
            vocal_16k = librosa.resample(vocal_stem_mono, orig_sr=sr, target_sr=vad_sr).astype(np.float32)
        else:
            vocal_16k = vocal_stem_mono

        wav_tensor = torch.from_numpy(vocal_16k)
        timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=vad_sr)
        vocal_regions = [
            {"start": float(ts["start"]) / vad_sr, "end": float(ts["end"]) / vad_sr}
            for ts in timestamps
        ]

    import os
    os.unlink(tmp_path)

    return {"stems": stems, "vocal_regions": vocal_regions}
'''


def deploy_modal_app() -> None:
    """Deploy the robomixer-gpu Modal app."""
    _ensure_modal()
    import tempfile
    import subprocess

    # Write the Modal app code to a temp file and deploy
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(MODAL_APP_CODE)
        tmp_path = f.name

    logger.info("Deploying robomixer-gpu Modal app...")
    result = subprocess.run(
        ["modal", "deploy", tmp_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Modal deploy failed:\n{result.stderr}")

    import os
    os.unlink(tmp_path)
    logger.info("Modal app deployed successfully")
