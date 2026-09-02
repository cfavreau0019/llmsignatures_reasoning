"""Summarise a trajectory into a flat feature dict.

Per the reference note, each chain is described by

  * the global-shape scalars  S, E, sigma
  * the mean / max / std of its curvatures
  * the mean / max / std of its torsions

plus book-keeping (`n_points`, and the fraction of interior vertices excluded by
the numerical guards for each estimator).
"""
from __future__ import annotations

import numpy as np

from . import geometry
from .guards import Guards

FEATURE_KEYS = (
    "n_points",
    "path_length",          # S
    "dirichlet_energy",     # E
    "straightness",         # sigma
    "curvature_mean",
    "curvature_max",
    "curvature_std",
    "torsion_mean",
    "torsion_max",
    "torsion_std",
    "curvature_excluded_frac",
    "torsion_excluded_frac",
)


def _agg(values: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, max, std, excluded_fraction) over the valid entries."""
    total = valid.size
    kept = values[valid] if total else values
    kept = kept[np.isfinite(kept)]
    excluded = 1.0 - (kept.size / total) if total else float("nan")
    if kept.size == 0:
        return float("nan"), float("nan"), float("nan"), excluded
    return float(kept.mean()), float(kept.max()), float(kept.std()), excluded


def summarize_trajectory(X, guards: Guards | None = None) -> dict[str, float]:
    """Compute :data:`FEATURE_KEYS` for one trajectory ``X`` of shape (n, h)."""
    guards = guards or Guards()
    X = np.asarray(X, dtype=np.float64)

    kappa, kappa_valid = geometry.curvature(X, guards)
    tau, tau_valid = geometry.torsion(X, guards)

    k_mean, k_max, k_std, k_excl = _agg(kappa, kappa_valid)
    t_mean, t_max, t_std, t_excl = _agg(tau, tau_valid)

    return {
        "n_points": int(X.shape[0]),
        "path_length": geometry.path_length(X),
        "dirichlet_energy": geometry.dirichlet_energy(X),
        "straightness": geometry.straightness(X),
        "curvature_mean": k_mean,
        "curvature_max": k_max,
        "curvature_std": k_std,
        "torsion_mean": t_mean,
        "torsion_max": t_max,
        "torsion_std": t_std,
        "curvature_excluded_frac": k_excl,
        "torsion_excluded_frac": t_excl,
    }


def summarize_many(trajectories: dict[str, np.ndarray], guards: Guards | None = None) -> dict[str, float]:
    """Summarise several named trajectories, prefixing keys with the name.

    e.g. ``{"token": X_tok, "layer": X_lay}`` ->
    ``{"token_path_length": ..., "layer_path_length": ...}``.
    """
    out: dict[str, float] = {}
    for name, X in trajectories.items():
        for k, v in summarize_trajectory(X, guards).items():
            out[f"{name}_{k}"] = v
    return out
