"""Analytic checks for the discrete curve-geometry estimators.

A discretely sampled circle of radius R has curvature -> 1/R and zero torsion.
A helix x(t) = (R cos t, R sin t, c t) has curvature R/(R^2+c^2) and torsion
c/(R^2+c^2), both constant.
"""
import numpy as np
import pytest

from reasoning_trajectories.trajectories import (
    curvature,
    dirichlet_energy,
    path_length,
    straightness,
    summarize_trajectory,
    torsion,
)


def test_straight_line():
    X = np.linspace(0, 1, 20)[:, None] * np.array([[1.0, 2.0, -3.0, 0.5]])
    assert straightness(X) == pytest.approx(1.0, abs=1e-12)
    kappa, valid = curvature(X)
    assert valid.all()
    assert np.allclose(kappa, 0.0, atol=1e-5)  # arccos near 1 -> ~1e-7 float noise
    tau, tvalid = torsion(X)
    assert np.allclose(tau[tvalid], 0.0, atol=1e-6)


def test_path_length_and_energy():
    X = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])  # steps of length 3 and 4
    assert path_length(X) == pytest.approx(7.0)
    assert dirichlet_energy(X) == pytest.approx(0.5 * (9 + 16))
    assert straightness(X) == pytest.approx(5.0 / 7.0)  # ||end-start||=5, S=7


def test_circle_curvature_and_zero_torsion():
    R = 2.0
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    X = np.c_[R * np.cos(t), R * np.sin(t), np.zeros_like(t)]
    kappa, valid = curvature(X)
    assert valid.mean() > 0.999
    assert np.median(kappa[valid]) == pytest.approx(1.0 / R, rel=2e-3)
    tau, tvalid = torsion(X)
    # planar curve: torsion identically ~0 (guard drops the exactly-degenerate ones)
    assert np.nanmax(np.abs(tau[tvalid])) < 1e-3


def test_helix_curvature_and_torsion():
    R, c = 1.5, 0.7
    t = np.linspace(0, 6 * np.pi, 1500)
    X = np.c_[R * np.cos(t), R * np.sin(t), c * t]
    exp_kappa = R / (R**2 + c**2)
    exp_tau = c / (R**2 + c**2)
    kappa, kv = curvature(X)
    tau, tv = torsion(X)
    assert np.median(kappa[kv]) == pytest.approx(exp_kappa, rel=5e-3)
    assert np.median(tau[tv]) == pytest.approx(exp_tau, rel=5e-2)


def test_torsion_invariant_to_sampling_density():
    # Same helix, sampled at two densities -> same torsion estimate
    # ("all powers of the step size cancel", reference note).
    R, c = 1.0, 0.5
    exp_tau = c / (R**2 + c**2)
    for n in (400, 1600):
        t = np.linspace(0, 8 * np.pi, n)
        X = np.c_[R * np.cos(t), R * np.sin(t), c * t]
        tau, tv = torsion(X)
        assert np.median(tau[tv]) == pytest.approx(exp_tau, rel=5e-2)


def test_summarize_keys_and_guard_report():
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.standard_normal((120, 8)), axis=0)
    feats = summarize_trajectory(X)
    for k in ("path_length", "dirichlet_energy", "straightness",
              "curvature_mean", "curvature_max", "curvature_std",
              "torsion_mean", "torsion_max", "torsion_std",
              "curvature_excluded_frac", "torsion_excluded_frac"):
        assert k in feats
    assert 0.0 <= feats["straightness"] <= 1.0
    assert feats["curvature_excluded_frac"] < 1e-3


def test_short_trajectories_degrade_gracefully():
    assert curvature(np.zeros((2, 4)))[0].size == 0
    assert torsion(np.zeros((3, 4)))[0].size == 0
    feats = summarize_trajectory(np.zeros((1, 4)))
    assert feats["n_points"] == 1
