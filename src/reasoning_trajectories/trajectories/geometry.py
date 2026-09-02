r"""Discrete curve geometry for hidden-state trajectories.

A trajectory is an ordered list of points ``X = (x_1, ..., x_n)`` stored as an
array of shape ``(n, h)``.  Everything here is computed in the *full* hidden
dimension ``h`` (the quantities depend only on distances and inner products, so
no dimensionality reduction is used) and in ``float64``.

Quantities (see the reference note):

  step vectors        d_i  = x_{i+1} - x_i
  path length         S(X) = sum_i ||d_i||
  Dirichlet energy    E(X) = 1/2 sum_i ||d_i||^2
  straightness        sigma(X) = ||x_n - x_1|| / S(X)          (1 == perfectly straight)

  curvature at x_i    kappa_i = theta_i / sbar_i
      theta_i = arccos( <d_{i-1}, d_i> / (||d_{i-1}|| ||d_i||) )
      sbar_i  = 1/2 (||d_{i-1}|| + ||d_i||)

  torsion at x_i (needs x_{i-1..i+2}):
      v1 = 1/2 (x_{i+1} - x_{i-1})
      v2 = x_{i+1} - 2 x_i + x_{i-1}
      v3 = x_{i+2} - 3 x_{i+1} + 3 x_i - x_{i-1}
      |tau_i| = sqrt(det G(v1,v2,v3)) / ||v1 ^ v2||^2
      ||v1 ^ v2||^2 = ||v1||^2 ||v2||^2 - <v1, v2>^2

Each estimator returns ``(values, valid_mask)``.  Entries failing a
:class:`~reasoning_trajectories.trajectories.guards.Guards` test are returned as
``nan`` with ``valid_mask == False`` so callers can report the excluded fraction.
"""
from __future__ import annotations

import numpy as np

from .guards import Guards

__all__ = [
    "steps",
    "step_norms",
    "path_length",
    "dirichlet_energy",
    "straightness",
    "curvature",
    "torsion",
]


def _as2d(X) -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"trajectory must be 2-D (n, h); got shape {A.shape}")
    return A


def steps(X) -> np.ndarray:
    """Step vectors d_i = x_{i+1} - x_i, shape (n-1, h)."""
    return np.diff(_as2d(X), axis=0)


def step_norms(X) -> np.ndarray:
    """Euclidean length of each step, shape (n-1,)."""
    return np.linalg.norm(steps(X), axis=1)


# --------------------------------------------------------------------------- #
# global shape
# --------------------------------------------------------------------------- #
def path_length(X) -> float:
    return float(step_norms(X).sum())


def dirichlet_energy(X) -> float:
    return float(0.5 * np.sum(step_norms(X) ** 2))


def straightness(X) -> float:
    A = _as2d(X)
    S = path_length(A)
    if S == 0.0:
        return float("nan")
    return float(np.linalg.norm(A[-1] - A[0]) / S)


# --------------------------------------------------------------------------- #
# curvature
# --------------------------------------------------------------------------- #
def curvature(X, guards: Guards | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Discrete curvature kappa_i at interior vertices x_2 .. x_{n-1}.

    Returns ``(kappa, valid)`` each of length ``n - 2`` (empty if ``n < 3``).
    """
    guards = guards or Guards()
    A = _as2d(X)
    n = A.shape[0]
    if n < 3:
        return np.empty(0), np.empty(0, dtype=bool)

    D = np.diff(A, axis=0)                     # (n-1, h)
    dn = np.linalg.norm(D, axis=1)             # (n-1,)
    a, b = D[:-1], D[1:]                       # incoming / outgoing, (n-2, h)
    na, nb = dn[:-1], dn[1:]

    thr = max(guards.step_abs_tol, guards.step_rel_tol * _median_positive(dn))
    valid = (na > thr) & (nb > thr)

    denom = np.where(valid, na * nb, 1.0)
    cos = np.einsum("ij,ij->i", a, b) / denom
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    sbar = 0.5 * (na + nb)

    kappa = np.full(n - 2, np.nan)
    kappa[valid] = theta[valid] / sbar[valid]
    return kappa, valid


# --------------------------------------------------------------------------- #
# torsion
# --------------------------------------------------------------------------- #
def torsion(X, guards: Guards | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned discrete torsion |tau_i| at vertices x_2 .. x_{n-2}.

    Uses four consecutive points (x_{i-1}, x_i, x_{i+1}, x_{i+2}).
    Returns ``(tau, valid)`` each of length ``n - 3`` (empty if ``n < 4``).
    """
    guards = guards or Guards()
    A = _as2d(X)
    n = A.shape[0]
    if n < 4:
        return np.empty(0), np.empty(0, dtype=bool)

    xm1 = A[0 : n - 3]
    x0 = A[1 : n - 2]
    xp1 = A[2 : n - 1]
    xp2 = A[3 : n]

    v1 = 0.5 * (xp1 - xm1)
    v2 = xp1 - 2.0 * x0 + xm1
    v3 = xp2 - 3.0 * xp1 + 3.0 * x0 - xm1

    g11 = np.einsum("ij,ij->i", v1, v1)
    g22 = np.einsum("ij,ij->i", v2, v2)
    g33 = np.einsum("ij,ij->i", v3, v3)
    g12 = np.einsum("ij,ij->i", v1, v2)
    g13 = np.einsum("ij,ij->i", v1, v3)
    g23 = np.einsum("ij,ij->i", v2, v3)

    det_g = (
        g11 * (g22 * g33 - g23 * g23)
        - g12 * (g12 * g33 - g23 * g13)
        + g13 * (g12 * g23 - g22 * g13)
    )
    wedge2 = g11 * g22 - g12 * g12  # ||v1 ^ v2||^2

    # degenerate-plane guard: sin^2(angle(v1,v2)) = wedge2 / (||v1||^2 ||v2||^2)
    scale = g11 * g22
    with np.errstate(divide="ignore", invalid="ignore"):
        sin2 = np.where(scale > 0, wedge2 / scale, 0.0)
    valid = (wedge2 > 0) & (sin2 > guards.plane_sin2_tol)

    tau = np.full(n - 3, np.nan)
    num = np.sqrt(np.clip(det_g[valid], 0.0, None))
    tau[valid] = num / wedge2[valid]
    return tau, valid


# --------------------------------------------------------------------------- #
def _median_positive(x: np.ndarray) -> float:
    pos = x[x > 0]
    return float(np.median(pos)) if pos.size else 0.0
