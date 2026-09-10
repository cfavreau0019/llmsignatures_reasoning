"""reasoning_trajectories -- hidden-state trajectories of LLM reasoning chains.

Typical use::

    from reasoning_trajectories import Config, run_pipeline
    manifest = run_pipeline(Config.load("configs/default.yaml"))

The geometry layer is independent of extraction and can be used on any array::

    from reasoning_trajectories.trajectories import summarize_trajectory
    feats = summarize_trajectory(X)          # X: (n_points, hidden_dim)
"""
from .config import Config
from .env import load_env
from .reasoning import ReasoningChain
from .trajectories import (
    Guards,
    LayerTrajectory,
    TokenTrajectory,
    curvature,
    summarize_trajectory,
    torsion,
)

# Load a project .env (if present) once, so os.environ is populated before any
# code reads NDIF_API_KEY / HF_TOKEN. Never raises.
try:
    load_env()
except Exception:  # pragma: no cover - defensive
    pass

__all__ = [
    "Config",
    "run_pipeline",
    "load_env",
    "ReasoningChain",
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
