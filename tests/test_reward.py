"""Tests for continuous reward computation — log(speedup) + Nsight bonus."""
import math
import pytest

from openenv_env.reward import compute_reward, trloo_post_process


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_fail():
    assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_no_speedup():
    """speedup=1.0 → log(1.0) = 0.0"""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
    assert r == pytest.approx(0.0, abs=1e-6)


def test_modest_speedup():
    """speedup=1.5 → log(1.5) ≈ 0.405"""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=0.9)
    assert r == pytest.approx(math.log(1.5), abs=1e-4)


def test_large_speedup():
    """speedup=3.0 → log(3.0) ≈ 1.099"""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=3.0, speedup_vs_compile=1.0)
    assert r == pytest.approx(math.log(3.0), abs=1e-4)


def test_slower_than_baseline():
    """speedup=0.5 → log(0.5) ≈ -0.693 (negative but not -1.0 — kernel IS correct)"""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3)
    assert r == pytest.approx(math.log(0.5), abs=1e-4)


def test_very_slow_clamped():
    """speedup near 0 → clamped to log(0.1) = -2.302"""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.01, speedup_vs_compile=0)
    assert r == pytest.approx(math.log(0.1), abs=1e-4)


def test_nsight_bonus():
    """Nsight metrics add bonus on top of log(speedup)."""
    base = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
    with_nsight = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=0.8, mem_coalescing=0.9, warp_efficiency=0.7,
    )
    expected_bonus = 0.4 * 0.8 + 0.3 * 0.9 + 0.2 * 0.7  # 0.32 + 0.27 + 0.14 = 0.73
    assert with_nsight == pytest.approx(base + expected_bonus, abs=1e-4)


def test_nsight_clamped():
    """Nsight values clamped to [0, 1]."""
    r = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=1.5, mem_coalescing=-0.1, warp_efficiency=0.5,
    )
    expected = math.log(2.0) + 0.4 * 1.0 + 0.3 * 0.0 + 0.2 * 0.5
    assert r == pytest.approx(expected, abs=1e-4)


def test_trloo_post_process_g4():
    """TRLOO scales by N/(N-1) = 4/3 for G=4."""
    advantages = [0.5, -0.3, 1.2, -0.8]
    scaled = trloo_post_process(advantages, n=4)
    scale = 4 / 3
    for orig, result in zip(advantages, scaled):
        assert result == pytest.approx(orig * scale, abs=1e-6)


def test_trloo_post_process_g1():
    """TRLOO with N=1 returns unchanged."""
    advantages = [0.5]
    assert trloo_post_process(advantages, n=1) == advantages


def test_trloo_post_process_g2():
    """TRLOO scales by 2/1 = 2.0 for G=2."""
    advantages = [0.3, -0.3]
    scaled = trloo_post_process(advantages, n=2)
    assert scaled[0] == pytest.approx(0.6, abs=1e-6)
    assert scaled[1] == pytest.approx(-0.6, abs=1e-6)
