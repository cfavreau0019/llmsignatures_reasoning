"""Trajectory containers and builders.

Two trajectories are built from the per-layer hidden states of one reasoning
chain (a mapping ``layer_index -> array[n_positions, h]`` where layer 0 is the
token-embedding output and layer L is the final residual stream):

  * :class:`TokenTrajectory` -- follows the text.  At a fixed anchor layer
    ``l*`` (two thirds of network depth) it takes the hidden state of every
    chain-token position:  ``X_tok = (x_1, ..., x_n)``.

  * :class:`LayerTrajectory` -- follows depth.  For every layer it averages the
    hidden states of all chain-token positions into one point:
    ``X_lay = (xbar_0, ..., xbar_L)``.

``span`` is the slice of positions that count as "the reasoning chain"
(completion-only or the full prompt+completion), applied before either build.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from . import features as _features
from .guards import Guards


@dataclass
class TokenTrajectory:
    points: np.ndarray                 # (n, h)
    layer: int                         # anchor layer l*
    token_strings: list[str] | None = None
    meta: dict = field(default_factory=dict)

    def features(self, guards: Guards | None = None) -> dict[str, float]:
        return _features.summarize_trajectory(self.points, guards)


@dataclass
class LayerTrajectory:
    points: np.ndarray                 # (L+1, h)
    layer_indices: np.ndarray          # (L+1,)
    meta: dict = field(default_factory=dict)

    def features(self, guards: Guards | None = None) -> dict[str, float]:
        return _features.summarize_trajectory(self.points, guards)


# --------------------------------------------------------------------------- #
def _resolve_span(span, n_positions: int) -> slice:
    if span is None or span == "full" or span == slice(None):
        return slice(0, n_positions)
    if isinstance(span, slice):
        return span
    start, stop = span
    return slice(int(start), int(stop) if stop is not None else n_positions)


def build_token_trajectory(
    hidden_by_layer: Mapping[int, np.ndarray],
    star_layer: int,
    span=None,
    token_strings: list[str] | None = None,
) -> TokenTrajectory:
    H = np.asarray(hidden_by_layer[star_layer], dtype=np.float64)  # (n_pos, h)
    sl = _resolve_span(span, H.shape[0])
    pts = H[sl]
    toks = token_strings[sl] if token_strings is not None else None
    return TokenTrajectory(
        points=pts,
        layer=int(star_layer),
        token_strings=list(toks) if toks is not None else None,
        meta={"span": [sl.start, sl.stop], "n_positions_total": int(H.shape[0])},
    )


def build_layer_trajectory(
    hidden_by_layer: Mapping[int, np.ndarray],
    span=None,
) -> LayerTrajectory:
    layers = sorted(hidden_by_layer)
    any_layer = np.asarray(hidden_by_layer[layers[0]])
    sl = _resolve_span(span, any_layer.shape[0])
    pts = np.stack(
        [np.asarray(hidden_by_layer[l], dtype=np.float64)[sl].mean(axis=0) for l in layers],
        axis=0,
    )
    return LayerTrajectory(
        points=pts,
        layer_indices=np.asarray(layers, dtype=int),
        meta={"span": [sl.start, sl.stop]},
    )


def build_layer_trajectory_from_means(
    mean_by_layer: Mapping[int, np.ndarray],
) -> LayerTrajectory:
    """Build the depth trajectory when position-means were already reduced
    inside the model trace (one vector per layer)."""
    layers = sorted(mean_by_layer)
    pts = np.stack([np.asarray(mean_by_layer[l], dtype=np.float64).reshape(-1) for l in layers], axis=0)
    return LayerTrajectory(points=pts, layer_indices=np.asarray(layers, dtype=int))
