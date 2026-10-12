"""Unit tests for psi(), gini() and ks_curve(): synthetic data only."""
import numpy as np
import pytest
from metrics import gini, ks_curve, psi, rag


def test_psi_identical_distributions_is_zero():
    e = np.array([0.2, 0.3, 0.5])
    assert psi(e, e) == pytest.approx(0.0, abs=1e-12)


def test_psi_known_value():
    # (0.6-0.5) ln(0.6/0.5) + (0.4-0.5) ln(0.4/0.5) = 0.040546
    assert psi([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.040546, abs=1e-4)


def test_psi_is_non_negative_and_symmetric():
    rng = np.random.default_rng(42)
    a, b = rng.dirichlet(np.ones(9)), rng.dirichlet(np.ones(9))
    assert psi(a, b) >= 0 and psi(a, b) == pytest.approx(psi(b, a), rel=1e-6)


def test_psi_handles_empty_bins():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


def test_gini_perfect_reversed_and_constant():
    y = np.array([0, 0, 0, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0)
    assert gini(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1])) == pytest.approx(-1.0)
    assert gini(y, np.full(5, 0.5)) == pytest.approx(0.0)


def test_gini_random_scores_near_zero():
    rng = np.random.default_rng(42)
    y = rng.random(20000) < 0.05
    assert abs(gini(y, rng.random(20000))) < 0.05


def test_ks_is_one_for_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    _, _, ks = ks_curve(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert ks == pytest.approx(1.0)


def test_rag_thresholds():
    assert rag(0.05, 0.10, 0.25) == "Green"
    assert rag(0.10, 0.10, 0.25) == "Amber"
    assert rag(0.26, 0.10, 0.25) == "Red"
