"""Rule-based transition scorer.

Provides immediate value before ML models are trained. Scores transitions based on:
- Camelot harmonic compatibility
- BPM matching (within configurable threshold)
- Energy flow continuity
- Vocal clash avoidance
- Phrase alignment quality
"""

from __future__ import annotations

import numpy as np

from robomixer.config import settings
from robomixer.models.song import SongAnalysis
from robomixer.models.transition import TransitionPoint, TransitionScore, TransitionType

# Camelot wheel: compatible keys are same code, ±1 in number, or parallel (A<->B)
# e.g., "5A" is compatible with "4A", "5A", "6A", "5B"
CAMELOT_CODES = [
    "1A",
    "2A",
    "3A",
    "4A",
    "5A",
    "6A",
    "7A",
    "8A",
    "9A",
    "10A",
    "11A",
    "12A",
    "1B",
    "2B",
    "3B",
    "4B",
    "5B",
    "6B",
    "7B",
    "8B",
    "9B",
    "10B",
    "11B",
    "12B",
]


def camelot_distance(code_a: str, code_b: str) -> int:
    """Compute the Camelot wheel distance between two keys.

    Returns 0 for same key, 1 for adjacent (±1 or parallel), higher for farther.
    """
    if not code_a or not code_b:
        return 7  # unknown key, neutral score

    num_a, letter_a = int(code_a[:-1]), code_a[-1]
    num_b, letter_b = int(code_b[:-1]), code_b[-1]

    if code_a == code_b:
        return 0

    # Parallel key (same number, different letter)
    if num_a == num_b and letter_a != letter_b:
        return 1

    # Adjacent number, same letter
    if letter_a == letter_b:
        diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
        return diff

    # Different letter and different number
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    return diff + 1


def score_harmonic(analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> float:
    """Score harmonic compatibility using Camelot wheel."""
    dist = camelot_distance(analysis_a.camelot_code, analysis_b.camelot_code)
    if dist == 0:
        return 1.0
    elif dist == 1:
        return 0.85
    elif dist == 2:
        return 0.5
    else:
        return max(0.0, 1.0 - dist * 0.2)


def score_tempo(analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> float:
    """Score BPM compatibility. Accounts for half/double time."""
    if analysis_a.bpm == 0 or analysis_b.bpm == 0:
        return 0.5

    bpm_a, bpm_b = analysis_a.bpm, analysis_b.bpm

    # Check direct match and half/double time — use symmetric percentage diff
    candidates = [bpm_b, bpm_b * 2, bpm_b / 2]
    best_diff = min(abs(c - bpm_a) / max(c, bpm_a) for c in candidates)

    threshold = settings.max_bpm_diff_pct
    if best_diff <= threshold:
        return 1.0 - (best_diff / threshold)
    return 0.0


def score_energy(exit_point: TransitionPoint, entry_point: TransitionPoint) -> float:
    """Score energy flow continuity between exit and entry points."""
    diff = abs(exit_point.local_energy - entry_point.local_energy)
    return max(0.0, 1.0 - diff)


def score_spectral(exit_point: TransitionPoint, entry_point: TransitionPoint) -> float:
    """Score timbral compatibility via cosine similarity on MFCC spectral profiles.

    Returns 0.5 (neutral) when either profile is empty or zero-norm.
    Maps cosine similarity from [-1, 1] to [0, 1].
    """
    if not exit_point.spectral_profile or not entry_point.spectral_profile:
        return 0.5

    a = np.asarray(exit_point.spectral_profile, dtype=np.float64)
    b = np.asarray(entry_point.spectral_profile, dtype=np.float64)

    # Truncate to the shorter length if mismatched
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5

    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Map [-1, 1] -> [0, 1]
    return (cosine_sim + 1.0) / 2.0


def score_vocal(exit_point: TransitionPoint, entry_point: TransitionPoint) -> float:
    """Penalize vocal clash (both tracks have vocals at transition point)."""
    if exit_point.has_vocals and entry_point.has_vocals:
        return settings.vocal_clash_penalty
    return 1.0


def score_structural(exit_point: TransitionPoint, entry_point: TransitionPoint) -> float:
    """Score phrase alignment and structural suitability."""
    score = 0.5

    if exit_point.phrase_aligned:
        score += 0.25
    if entry_point.phrase_aligned:
        score += 0.25

    # Prefer exits from outro/break, entries from intro/break
    good_exit_segments = {"outro", "break", "inst"}
    good_entry_segments = {"intro", "break", "verse"}

    if exit_point.current_segment in good_exit_segments:
        score = min(1.0, score + 0.1)
    if entry_point.current_segment in good_entry_segments:
        score = min(1.0, score + 0.1)

    return min(1.0, score)


def suggest_technique(
    analysis_a: SongAnalysis,
    analysis_b: SongAnalysis,
    exit_point: TransitionPoint,
    entry_point: TransitionPoint,
) -> tuple[TransitionType, int]:
    """Suggest a transition technique and overlap length based on exit/entry characteristics.

    Returns (TransitionType, overlap_beats).
    """
    bpm_a, bpm_b = analysis_a.bpm, analysis_b.bpm
    bpm_close = (
        bpm_a > 0
        and bpm_b > 0
        and abs(bpm_a - bpm_b) / max(bpm_a, bpm_b) <= settings.max_bpm_diff_pct
    )

    energy_diff = abs(exit_point.local_energy - entry_point.local_energy)
    both_low_energy = exit_point.local_energy < 0.3 and entry_point.local_energy < 0.3
    breakdown_segments = {"break", "outro", "intro", "inst"}
    exit_in_breakdown = exit_point.current_segment in breakdown_segments
    entry_in_breakdown = entry_point.current_segment in breakdown_segments

    # Long blend: both tracks in low-energy/breakdown sections
    if both_low_energy and exit_in_breakdown and entry_in_breakdown:
        return TransitionType.LONG_BLEND, 64

    # Echo out: exiting a vocal section
    if exit_point.has_vocals and not entry_point.has_vocals:
        return TransitionType.ECHO_OUT, 8

    # Cut: large BPM difference or high energy contrast
    if not bpm_close or energy_diff > 0.6:
        return TransitionType.CUT, 1

    # Filter sweep: moderate energy difference needing a smooth transition
    if 0.3 < energy_diff <= 0.6:
        return TransitionType.FILTER_SWEEP, 16

    # Beatmix: BPM is close (default for compatible tracks)
    return TransitionType.BEATMIX, 32


def score_transition(
    analysis_a: SongAnalysis,
    analysis_b: SongAnalysis,
    exit_point: TransitionPoint,
    entry_point: TransitionPoint,
) -> TransitionScore:
    """Compute full heuristic transition score between two songs at given cue points."""
    technique, overlap = suggest_technique(analysis_a, analysis_b, exit_point, entry_point)

    result = TransitionScore(
        exit_song_id=analysis_a.song_id,
        entry_song_id=analysis_b.song_id,
        exit_timestamp=exit_point.timestamp,
        entry_timestamp=entry_point.timestamp,
        harmonic_score=score_harmonic(analysis_a, analysis_b),
        tempo_score=score_tempo(analysis_a, analysis_b),
        energy_score=score_energy(exit_point, entry_point),
        vocal_score=score_vocal(exit_point, entry_point),
        structural_score=score_structural(exit_point, entry_point),
        spectral_score=score_spectral(exit_point, entry_point),
        suggested_technique=technique,
        overlap_beats=overlap,
    )
    result.compute_overall()
    return result
