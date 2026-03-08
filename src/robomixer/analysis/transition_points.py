"""Extract candidate transition points (exit and entry cues) from analyzed songs.

Exit points: phrase boundaries in the last portion of the song where energy is declining
             or stable-low, vocals have stopped, and the segment is outro/break/instrumental.

Entry points: phrase boundaries in the first portion where energy is rising or stable-low,
              before vocals begin, and the segment is intro/break/first verse onset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from robomixer.config import settings
from robomixer.models.song import SongAnalysis
from robomixer.models.transition import EnergyDirection, PointType, TransitionPoint


def _context_window(timestamp: float, beat_times: list[float], n_beats: int) -> tuple[float, float]:
    """Compute a context window of n_beats centred on timestamp."""
    if not beat_times:
        return timestamp, timestamp
    arr = np.array(beat_times)
    centre_idx = int(np.argmin(np.abs(arr - timestamp)))
    start_idx = max(0, centre_idx - n_beats // 2)
    end_idx = min(len(arr) - 1, centre_idx + n_beats // 2)
    return float(arr[start_idx]), float(arr[end_idx])


def _beats_until_boundary(timestamp: float, analysis: SongAnalysis) -> int:
    """Count beats from timestamp to the nearest segment boundary."""
    segment = analysis.segment_at(timestamp)
    if segment is None or not analysis.beat_times:
        return 0
    boundary = segment.end
    arr = np.array(analysis.beat_times)
    beats_after = arr[(arr >= timestamp) & (arr <= boundary)]
    return len(beats_after)


def _compute_energy_direction(
    rms_curve: NDArray[np.float32], timestamp: float, sr: int, hop_length: int
) -> EnergyDirection:
    """Determine if energy is rising, falling, or stable around a timestamp."""
    frame_idx = int(timestamp * sr / hop_length)
    # Look at a window of ~1 second on each side
    window = max(1, sr // hop_length)
    start = max(0, frame_idx - window)
    end = min(len(rms_curve), frame_idx + window)
    if end - start < 2:
        return EnergyDirection.STABLE

    segment = rms_curve[start:end]
    slope = np.polyfit(np.arange(len(segment)), segment, 1)[0]

    threshold = 0.001
    if slope > threshold:
        return EnergyDirection.RISING
    elif slope < -threshold:
        return EnergyDirection.FALLING
    return EnergyDirection.STABLE


def _compute_breakdown_score(
    rms_curve: NDArray[np.float32], timestamp: float, sr: int, hop_length: int
) -> float:
    """Score how stripped-back / open the mix is at this point (0=full, 1=silent)."""
    frame_idx = int(timestamp * sr / hop_length)
    # Use a window of ~2 seconds centred on the timestamp
    window = max(1, 2 * sr // hop_length)
    start = max(0, frame_idx - window // 2)
    end = min(len(rms_curve), frame_idx + window // 2)
    if end <= start:
        return 0.0

    local_energy = float(np.mean(rms_curve[start:end]))
    global_energy = float(np.mean(rms_curve))
    if global_energy == 0:
        return 0.0
    # Invert: low local energy relative to global = high breakdown score
    return float(np.clip(1.0 - (local_energy / global_energy), 0.0, 1.0))


def _compute_spectral_profile(
    audio: NDArray[np.float32], sr: int, timestamp: float, window_sec: float = 2.0
) -> list[float]:
    """Extract a compact MFCC summary around a timestamp."""
    import librosa

    start_sample = max(0, int((timestamp - window_sec / 2) * sr))
    end_sample = min(len(audio), int((timestamp + window_sec / 2) * sr))
    if end_sample - start_sample < sr // 4:
        return []

    segment = audio[start_sample:end_sample]
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    # Return the mean of each MFCC coefficient across the window
    return [float(x) for x in np.mean(mfcc, axis=1)]


def enrich_point(
    point: TransitionPoint,
    analysis: SongAnalysis,
    audio: NDArray[np.float32],
    sr: int,
    rms_curve: NDArray[np.float32] | None = None,
    hop_length: int = 512,
) -> TransitionPoint:
    """Populate all spectral/context fields on a TransitionPoint."""
    import librosa

    # Compute RMS if not provided
    if rms_curve is None:
        rms_curve = librosa.feature.rms(y=audio, hop_length=hop_length).squeeze()

    # Context window
    n_beats = settings.context_window_beats
    point.window_start, point.window_end = _context_window(
        point.timestamp, analysis.beat_times, n_beats
    )

    # Global features as local approximations
    point.local_bpm = analysis.bpm
    point.local_key = analysis.key

    # Local energy from RMS curve
    frame_idx = int(point.timestamp * sr / hop_length)
    frame_idx = min(frame_idx, len(rms_curve) - 1)
    point.local_energy = float(rms_curve[frame_idx]) if len(rms_curve) > 0 else 0.0

    # Energy direction
    point.energy_direction = _compute_energy_direction(rms_curve, point.timestamp, sr, hop_length)

    # Breakdown score
    point.breakdown_score = _compute_breakdown_score(rms_curve, point.timestamp, sr, hop_length)

    # Beats until phrase boundary
    point.beats_until_phrase_boundary = _beats_until_boundary(point.timestamp, analysis)

    # Spectral profile (MFCC summary)
    point.spectral_profile = _compute_spectral_profile(audio, sr, point.timestamp)

    return point


def extract_exit_points(
    analysis: SongAnalysis,
    audio: NDArray[np.float32] | None = None,
    sr: int = 0,
) -> list[TransitionPoint]:
    """Find candidate exit points in the last portion of a song."""
    if not analysis.beat_times:
        return []

    duration = analysis.beat_times[-1]
    exit_start = duration * settings.exit_region_pct

    points: list[TransitionPoint] = []
    rms_curve = None

    for downbeat in analysis.downbeat_times:
        if downbeat < exit_start:
            continue

        segment = analysis.segment_at(downbeat)
        if segment is None:
            continue

        has_vocals = analysis.has_vocals_at(downbeat)

        point = TransitionPoint(
            song_id=analysis.song_id,
            timestamp=downbeat,
            point_type=PointType.EXIT,
            current_segment=segment.label,
            has_vocals=has_vocals,
            phrase_aligned=True,
        )

        # Enrich with spectral features if audio is available
        if audio is not None and sr > 0:
            import librosa

            if rms_curve is None:
                rms_curve = librosa.feature.rms(y=audio).squeeze()
            point = enrich_point(point, analysis, audio, sr, rms_curve)

        points.append(point)

    return points


def extract_entry_points(
    analysis: SongAnalysis,
    audio: NDArray[np.float32] | None = None,
    sr: int = 0,
) -> list[TransitionPoint]:
    """Find candidate entry points in the first portion of a song."""
    if not analysis.beat_times:
        return []

    duration = analysis.beat_times[-1]
    entry_end = duration * settings.entry_region_pct

    points: list[TransitionPoint] = []
    rms_curve = None

    for downbeat in analysis.downbeat_times:
        if downbeat > entry_end:
            break

        segment = analysis.segment_at(downbeat)
        if segment is None:
            continue

        has_vocals = analysis.has_vocals_at(downbeat)

        point = TransitionPoint(
            song_id=analysis.song_id,
            timestamp=downbeat,
            point_type=PointType.ENTRY,
            current_segment=segment.label,
            has_vocals=has_vocals,
            phrase_aligned=True,
        )

        # Enrich with spectral features if audio is available
        if audio is not None and sr > 0:
            import librosa

            if rms_curve is None:
                rms_curve = librosa.feature.rms(y=audio).squeeze()
            point = enrich_point(point, analysis, audio, sr, rms_curve)

        points.append(point)

    return points
