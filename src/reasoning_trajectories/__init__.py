"""reasoning_trajectories -- hidden-state trajectories of LLM reasoning chains.

Typical use::

    from reasoning_trajectories import Config, run_pipeline
    manifest = run_pipeline(Config.load("configs/default.yaml"))

The geometry layer is independent of extraction and can be used on any array::

    from reasoning_trajectories.trajectories import summarize_trajectory
    feats = summarize_trajectory(X)          # X: (n_points, hidden_dim)
"""
from .config import Config
from .trajectories import (
    Guards,
    LayerTrajectory,
    TokenTrajectory,
    curvature,
    summarize_trajectory,
    torsion,
)

__all__ = [
    "Config",
    "run_pipeline",
    "Guards",
    "TokenTrajectory",
    "LayerTrajectory",
    "curvature",
    "torsion",
    "summarize_trajectory",
]


def run_pipeline(cfg):
    from .pipeline import run_pipeline as _run

    return _run(cfg)
