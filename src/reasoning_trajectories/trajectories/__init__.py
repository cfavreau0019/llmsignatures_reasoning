"""Hidden-state trajectory geometry.

See ``studies/hidden_state_trajectories/README.md`` for the mapping between the
functions here and the quantities in the reference note.
"""
from .guards import Guards
from .geometry import (
    curvature,
    dirichlet_energy,
    path_length,
    step_norms,
    steps,
    straightness,
    torsion,
)
from .features import FEATURE_KEYS, summarize_many, summarize_trajectory
from .trajectory import (
    LayerTrajectory,
    TokenTrajectory,
    build_layer_trajectory,
    build_layer_trajectory_from_means,
    build_token_trajectory,
)

__all__ = [
    "Guards",
    "steps",
    "step_norms",
    "path_length",
    "dirichlet_energy",
    "straightness",
    "curvature",
    "torsion",
    "FEATURE_KEYS",
    "summarize_trajectory",
    "summarize_many",
    "TokenTrajectory",
    "LayerTrajectory",
    "build_token_trajectory",
    "build_layer_trajectory",
    "build_layer_trajectory_from_means",
]
