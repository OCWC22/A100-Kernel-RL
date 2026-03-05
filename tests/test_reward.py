"""Tests for reward computation — all 4 discrete milestone levels."""
import pytest

from openenv_env.reward import compute_reward


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_fail():
    assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_no_speedup():
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9) == 1.0


def test_beats_eager():
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.10, speedup_vs_compile=0.9) == 2.0


def test_beats_both():
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.10, speedup_vs_compile=1.10) == 3.0


def test_threshold_boundary_eager():
    """Exactly 1.05 should NOT trigger speedup reward (need >1.05)."""
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.05, speedup_vs_compile=0) == 1.0


def test_threshold_boundary_compile():
    """Exactly 1.05 should NOT trigger top reward (need >1.05)."""
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.10, speedup_vs_compile=1.05) == 2.0


def test_just_above_threshold():
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.051, speedup_vs_compile=0) == 2.0
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=1.10, speedup_vs_compile=1.051) == 3.0


def test_negative_speedup():
    """Slower than baseline still gets reward=1 if correct."""
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3) == 1.0


def test_zero_speedup():
    assert compute_reward(compiled=True, correct=True, speedup_vs_eager=0, speedup_vs_compile=0) == 1.0
