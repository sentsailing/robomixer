"""Tests for the heuristic transition scorer."""

from uuid import uuid4

import pytest

from robomixer.models.song import SongAnalysis
from robomixer.models.transition import PointType, TransitionPoint, TransitionType
from robomixer.scoring.heuristic import (
    camelot_distance,
    score_spectral,
    score_tempo,
    score_transition,
    score_vocal,
    suggest_technique,
)


def _make_analysis(**kwargs) -> SongAnalysis:
    defaults = {"song_id": uuid4(), "bpm": 128.0, "camelot_code": "5A"}
    defaults.update(kwargs)
    return SongAnalysis(**defaults)


def _make_point(song_id=None, **kwargs) -> TransitionPoint:
    defaults = {
        "song_id": song_id or uuid4(),
        "timestamp": 180.0,
        "point_type": PointType.EXIT,
        "local_energy": 0.5,
        "has_vocals": False,
    }
    defaults.update(kwargs)
    return TransitionPoint(**defaults)


class TestCamelotDistance:
    def test_same_key(self):
        assert camelot_distance("5A", "5A") == 0

    def test_parallel_key(self):
        assert camelot_distance("5A", "5B") == 1

    def test_adjacent_same_letter(self):
        assert camelot_distance("5A", "6A") == 1
        assert camelot_distance("5A", "4A") == 1

    def test_wrap_around(self):
        assert camelot_distance("1A", "12A") == 1

    def test_far_keys(self):
        assert camelot_distance("1A", "7A") == 6

    def test_unknown_key_a(self):
        assert camelot_distance("", "5A") == 7

    def test_unknown_key_b(self):
        assert camelot_distance("5A", "") == 7

    def test_both_unknown(self):
        assert camelot_distance("", "") == 7

    def test_opposite_side_of_wheel(self):
        # 1A to 7A = 6 steps either way on a 12-element wheel
        assert camelot_distance("1A", "7A") == 6
        # But 1A to 8A = min(7, 5) = 5
        assert camelot_distance("1A", "8A") == 5

    def test_cross_letter_wrap(self):
        # 1A to 12B: diff = min(11, 1) = 1, +1 for letter change = 2
        assert camelot_distance("1A", "12B") == 2

    def test_symmetric(self):
        assert camelot_distance("3A", "8B") == camelot_distance("8B", "3A")

    def test_parallel_b_to_a(self):
        assert camelot_distance("7B", "7A") == 1


class TestTempoScoring:
    def test_same_bpm(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        assert score_tempo(a, b) == 1.0

    def test_close_bpm(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=130.0)
        assert score_tempo(a, b) > 0.5

    def test_double_time(self):
        a = _make_analysis(bpm=70.0)
        b = _make_analysis(bpm=140.0)
        assert score_tempo(a, b) == 1.0

    def test_incompatible_bpm(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=90.0)
        assert score_tempo(a, b) == 0.0

    def test_zero_bpm_returns_neutral(self):
        a = _make_analysis(bpm=0.0)
        b = _make_analysis(bpm=128.0)
        assert score_tempo(a, b) == 0.5

    def test_both_zero_bpm(self):
        a = _make_analysis(bpm=0.0)
        b = _make_analysis(bpm=0.0)
        assert score_tempo(a, b) == 0.5

    def test_half_time(self):
        a = _make_analysis(bpm=140.0)
        b = _make_analysis(bpm=70.0)
        assert score_tempo(a, b) == 1.0

    def test_tempo_symmetric(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=130.0)
        assert score_tempo(a, b) == score_tempo(b, a)


class TestVocalScoring:
    def test_no_vocals(self):
        exit_p = _make_point(has_vocals=False)
        entry_p = _make_point(has_vocals=False, point_type=PointType.ENTRY)
        assert score_vocal(exit_p, entry_p) == 1.0

    def test_vocal_clash(self):
        exit_p = _make_point(has_vocals=True)
        entry_p = _make_point(has_vocals=True, point_type=PointType.ENTRY)
        assert score_vocal(exit_p, entry_p) < 1.0


class TestSpectralScoring:
    def test_identical_profiles(self):
        profile = [1.0, 2.0, 3.0, 4.0, 5.0]
        exit_p = _make_point(spectral_profile=profile)
        entry_p = _make_point(spectral_profile=profile, point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 1.0

    def test_opposite_profiles(self):
        exit_p = _make_point(spectral_profile=[1.0, 0.0, 0.0])
        entry_p = _make_point(spectral_profile=[-1.0, 0.0, 0.0], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.0

    def test_orthogonal_profiles(self):
        exit_p = _make_point(spectral_profile=[1.0, 0.0])
        entry_p = _make_point(spectral_profile=[0.0, 1.0], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.5

    def test_empty_exit_profile(self):
        exit_p = _make_point(spectral_profile=[])
        entry_p = _make_point(spectral_profile=[1.0, 2.0], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.5

    def test_empty_entry_profile(self):
        exit_p = _make_point(spectral_profile=[1.0, 2.0])
        entry_p = _make_point(spectral_profile=[], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.5

    def test_both_empty_profiles(self):
        exit_p = _make_point(spectral_profile=[])
        entry_p = _make_point(spectral_profile=[], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.5

    def test_zero_norm_profile(self):
        exit_p = _make_point(spectral_profile=[0.0, 0.0, 0.0])
        entry_p = _make_point(spectral_profile=[1.0, 2.0, 3.0], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == 0.5

    def test_mismatched_lengths(self):
        """Truncates to shorter length; [1, 2] vs [1, 2] (truncated) -> identical -> 1.0."""
        exit_p = _make_point(spectral_profile=[1.0, 2.0])
        entry_p = _make_point(spectral_profile=[1.0, 2.0, 99.0], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == pytest.approx(1.0)

    def test_similar_profiles_high_score(self):
        exit_p = _make_point(spectral_profile=[1.0, 2.0, 3.0, 4.0])
        entry_p = _make_point(spectral_profile=[1.1, 2.1, 3.1, 4.1], point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) > 0.99

    def test_symmetric(self):
        p_a = [1.0, 3.0, -2.0, 5.0]
        p_b = [2.0, -1.0, 4.0, 0.5]
        exit_p = _make_point(spectral_profile=p_a)
        entry_p = _make_point(spectral_profile=p_b, point_type=PointType.ENTRY)
        exit_p2 = _make_point(spectral_profile=p_b)
        entry_p2 = _make_point(spectral_profile=p_a, point_type=PointType.ENTRY)
        assert score_spectral(exit_p, entry_p) == score_spectral(exit_p2, entry_p2)


class TestSuggestTechnique:
    def test_beatmix_close_bpm_similar_energy(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        exit_p = _make_point(local_energy=0.6, has_vocals=False)
        entry_p = _make_point(local_energy=0.5, has_vocals=False, point_type=PointType.ENTRY)
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.BEATMIX
        assert overlap == 32

    def test_cut_large_bpm_difference(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=90.0)
        exit_p = _make_point(local_energy=0.5, has_vocals=False)
        entry_p = _make_point(local_energy=0.5, has_vocals=False, point_type=PointType.ENTRY)
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.CUT
        assert overlap == 1

    def test_cut_high_energy_contrast(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        exit_p = _make_point(local_energy=0.9, has_vocals=False)
        entry_p = _make_point(local_energy=0.1, has_vocals=False, point_type=PointType.ENTRY)
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.CUT
        assert overlap == 1

    def test_echo_out_vocal_exit(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        exit_p = _make_point(local_energy=0.5, has_vocals=True)
        entry_p = _make_point(local_energy=0.5, has_vocals=False, point_type=PointType.ENTRY)
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.ECHO_OUT
        assert overlap == 8

    def test_long_blend_both_breakdown(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        exit_p = _make_point(
            local_energy=0.2,
            has_vocals=False,
            current_segment="break",
        )
        entry_p = _make_point(
            local_energy=0.2,
            has_vocals=False,
            current_segment="intro",
            point_type=PointType.ENTRY,
        )
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.LONG_BLEND
        assert overlap == 64

    def test_filter_sweep_moderate_energy_diff(self):
        a = _make_analysis(bpm=128.0)
        b = _make_analysis(bpm=128.0)
        exit_p = _make_point(local_energy=0.7, has_vocals=False)
        entry_p = _make_point(local_energy=0.3, has_vocals=False, point_type=PointType.ENTRY)
        technique, overlap = suggest_technique(a, b, exit_p, entry_p)
        assert technique == TransitionType.FILTER_SWEEP
        assert overlap == 16

    def test_score_transition_includes_technique(self):
        """score_transition should populate suggested_technique and overlap_beats."""
        a = _make_analysis(bpm=128.0, camelot_code="5A")
        b = _make_analysis(bpm=128.0, camelot_code="5A")
        exit_p = _make_point(song_id=a.song_id, local_energy=0.5, has_vocals=False)
        entry_p = _make_point(
            song_id=b.song_id,
            local_energy=0.5,
            has_vocals=False,
            point_type=PointType.ENTRY,
        )
        result = score_transition(a, b, exit_p, entry_p)
        assert result.suggested_technique == TransitionType.BEATMIX
        assert result.overlap_beats == 32


class TestFullTransitionScore:
    def test_perfect_match(self):
        a = _make_analysis(bpm=128.0, camelot_code="5A")
        b = _make_analysis(bpm=128.0, camelot_code="5A")
        exit_p = _make_point(song_id=a.song_id, local_energy=0.3, has_vocals=False)
        entry_p = _make_point(
            song_id=b.song_id,
            local_energy=0.3,
            has_vocals=False,
            point_type=PointType.ENTRY,
        )
        result = score_transition(a, b, exit_p, entry_p)
        assert result.overall_score > 0.7

    def test_spectral_score_uses_profiles(self):
        """When spectral profiles are provided, spectral_score should not be 0.5 placeholder."""
        a = _make_analysis(bpm=128.0, camelot_code="5A")
        b = _make_analysis(bpm=128.0, camelot_code="5A")
        profile = [1.0, 2.0, 3.0, 4.0, 5.0]
        exit_p = _make_point(song_id=a.song_id, local_energy=0.5, spectral_profile=profile)
        entry_p = _make_point(
            song_id=b.song_id,
            local_energy=0.5,
            spectral_profile=profile,
            point_type=PointType.ENTRY,
        )
        result = score_transition(a, b, exit_p, entry_p)
        assert result.spectral_score == 1.0

    def test_spectral_score_neutral_without_profiles(self):
        """Without spectral profiles, spectral_score should be 0.5 (neutral)."""
        a = _make_analysis(bpm=128.0, camelot_code="5A")
        b = _make_analysis(bpm=128.0, camelot_code="5A")
        exit_p = _make_point(song_id=a.song_id, local_energy=0.5)
        entry_p = _make_point(
            song_id=b.song_id,
            local_energy=0.5,
            point_type=PointType.ENTRY,
        )
        result = score_transition(a, b, exit_p, entry_p)
        assert result.spectral_score == 0.5
