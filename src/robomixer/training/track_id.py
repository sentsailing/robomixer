"""Track identification within DJ mixes.

Uses a combination of:
1. Audio fingerprinting (Chromaprint/AcoustID) with sliding window
2. External tracklists from 1001Tracklists
3. Spectral novelty / change point detection for transition boundaries

The hardest unsolved problem in the training pipeline.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import acoustid
import librosa
import numpy as np
import ruptures as rpt
import soundfile as sf

from robomixer.config import settings

logger = logging.getLogger(__name__)

# Minimum number of consecutive matching windows to consider a track identified
_MIN_CLUSTER_SIZE = 2


@dataclass
class _WindowMatch:
    """A single fingerprint match from one sliding window."""

    window_start_sec: float
    window_end_sec: float
    acoustid_id: str
    title: str = ""
    artist: str = ""
    score: float = 0.0


@dataclass
class IdentifiedTrack:
    """A track identified within a DJ mix."""

    mix_start_sec: float
    mix_end_sec: float
    title: str = ""
    artist: str = ""
    acoustid: str = ""
    confidence: float = 0.0


class TrackIdentifier:
    """Identifies individual tracks within a DJ mix audio."""

    def __init__(
        self,
        window_sec: float = 15.0,
        stride_sec: float = 5.0,
        acoustid_api_key: str = "",
    ) -> None:
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self._api_key = acoustid_api_key or settings.acoustid_api_key

    def identify_tracks(self, mix_audio: np.ndarray, sr: int) -> list[IdentifiedTrack]:
        """Run sliding-window fingerprinting over a mix to identify tracks.

        Slides a window over the audio, fingerprints each window via Chromaprint,
        queries AcoustID, then clusters consecutive windows that match the same
        recording into IdentifiedTrack objects.
        """
        matches = list(self._fingerprint_windows(mix_audio, sr))
        if not matches:
            return []

        return self._cluster_matches(matches)

    def detect_transitions(self, mix_audio: np.ndarray, sr: int) -> list[float]:
        """Detect transition boundaries using spectral novelty and change point detection.

        Computes a spectral novelty function (onset strength envelope), then uses
        the ruptures library with a kernel change point detection algorithm to find
        transition boundaries.

        Returns a sorted list of timestamps (seconds) where transitions occur.
        """
        # Compute onset strength envelope as our novelty function
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=mix_audio, sr=sr, hop_length=hop_length)

        # Smooth the envelope to focus on large-scale transitions, not individual beats
        # Use a large window (~4 seconds) to capture mix-level changes
        smooth_frames = max(1, int(4.0 * sr / hop_length))
        if smooth_frames % 2 == 0:
            smooth_frames += 1
        kernel = np.ones(smooth_frames) / smooth_frames
        onset_smooth = np.convolve(onset_env, kernel, mode="same")

        # Use ruptures kernel-based change point detection
        # Penalty controls sensitivity — higher = fewer change points
        signal = onset_smooth.reshape(-1, 1)
        algo = rpt.KernelCPD(kernel="rbf", min_size=int(10.0 * sr / hop_length)).fit(signal)

        # Use penalty-based detection (BIC-like)
        n_samples = len(signal)
        penalty = np.log(n_samples) * 2
        breakpoints = algo.predict(pen=penalty)

        # Convert frame indices to timestamps (seconds)
        # ruptures returns indices into the signal; last element is always len(signal)
        timestamps: list[float] = []
        for bp in breakpoints:
            if bp < len(onset_smooth):
                t = librosa.frames_to_time(bp, sr=sr, hop_length=hop_length)
                timestamps.append(float(t))

        return sorted(timestamps)

    def _fingerprint_windows(self, mix_audio: np.ndarray, sr: int) -> Iterator[_WindowMatch]:
        """Slide a window over the audio and fingerprint each chunk."""
        window_samples = int(self.window_sec * sr)
        stride_samples = int(self.stride_sec * sr)

        pos = 0
        while pos + window_samples <= len(mix_audio):
            chunk = mix_audio[pos : pos + window_samples]
            start_sec = pos / sr
            end_sec = start_sec + self.window_sec

            match = self._fingerprint_chunk(chunk, sr, start_sec, end_sec)
            if match is not None:
                yield match

            pos += stride_samples

    def _fingerprint_chunk(
        self,
        chunk: np.ndarray,
        sr: int,
        start_sec: float,
        end_sec: float,
    ) -> _WindowMatch | None:
        """Fingerprint a single audio chunk and query AcoustID."""
        try:
            # Write chunk to a temporary WAV file for chromaprint
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, chunk, sr)

                # Query AcoustID
                results = acoustid.match(self._api_key, tmp.name)

                best_score = 0.0
                best_match: _WindowMatch | None = None
                for score, recording_id, title, artist in results:
                    if score > best_score:
                        best_score = score
                        best_match = _WindowMatch(
                            window_start_sec=start_sec,
                            window_end_sec=end_sec,
                            acoustid_id=recording_id or "",
                            title=title or "",
                            artist=artist or "",
                            score=score,
                        )

                return best_match

        except Exception:
            logger.debug("Fingerprint failed for window %.1f-%.1f", start_sec, end_sec)
            return None

    def _cluster_matches(self, matches: list[_WindowMatch]) -> list[IdentifiedTrack]:
        """Cluster consecutive windows with the same AcoustID into tracks.

        Groups adjacent windows that matched the same recording, then merges
        each group into an IdentifiedTrack spanning the full matched region.
        """
        if not matches:
            return []

        clusters: list[list[_WindowMatch]] = []
        current_cluster: list[_WindowMatch] = [matches[0]]

        for m in matches[1:]:
            prev = current_cluster[-1]
            # Same track if same acoustid and windows are adjacent (or overlapping)
            if (
                m.acoustid_id == prev.acoustid_id
                and m.acoustid_id
                and m.window_start_sec <= prev.window_end_sec + self.stride_sec
            ):
                current_cluster.append(m)
            else:
                clusters.append(current_cluster)
                current_cluster = [m]
        clusters.append(current_cluster)

        tracks: list[IdentifiedTrack] = []
        for cluster in clusters:
            if len(cluster) < _MIN_CLUSTER_SIZE:
                continue
            # Use the most common title/artist across windows
            scores = [m.score for m in cluster]
            best_idx = int(np.argmax(scores))
            best = cluster[best_idx]
            avg_confidence = float(np.mean(scores))

            tracks.append(
                IdentifiedTrack(
                    mix_start_sec=cluster[0].window_start_sec,
                    mix_end_sec=cluster[-1].window_end_sec,
                    title=best.title,
                    artist=best.artist,
                    acoustid=best.acoustid_id,
                    confidence=avg_confidence,
                )
            )

        return tracks
