"""Numerical guards for curve-geometry estimators.

Two degeneracies are excluded before any statistic is computed, following the
reference note ("From trajectories to features"):

  * a *vanishing step* -- when ||d_i|| is ~0 there is no direction of travel,
    so the vertex it touches has no well-defined turning angle;
  * a *degenerate plane* -- when the first two finite-difference vectors are
    nearly parallel, ||v1 ^ v2||^2 ~ 0 and there is no plane against which to
    measure twisting, so torsion is undefined there.

Both are relative tests so the thresholds are scale-free. The excluded fraction
is reported per trajectory and is expected to be < 1e-3 in practice.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Guards:
    # step "vanishes" if ||d|| <= step_rel_tol * median(||d||)  (or <= step_abs_tol)
    step_rel_tol: float = 1e-6
    step_abs_tol: float = 1e-12
    # v1,v2 "parallel" if  ||v1 ^ v2||^2 / (||v1||^2 ||v2||^2) <= plane_sin2_tol
    plane_sin2_tol: float = 1e-10

    @classmethod
    def from_config(cls, geometry_cfg) -> "Guards":
        return cls(
            step_rel_tol=geometry_cfg.step_rel_tol,
            step_abs_tol=geometry_cfg.step_abs_tol,
            plane_sin2_tol=geometry_cfg.plane_sin2_tol,
        )
