"""Per-example extraction: text -> targeted hidden states -> trajectories."""
from __future__ import annotations

from ..datasets.base import Example
from ..trajectories import (
    build_layer_trajectory_from_means,
    build_token_trajectory,
)
from . import layers as layer_utils
from .model import ReasoningModel
from .record import ExtractionRecord


class Extractor:
    def __init__(self, model: ReasoningModel, cfg):
        self.model = model
        self.cfg = cfg  # ExtractionConfig
        self.star_layer = layer_utils.resolve_layer(cfg.star_layer, model.n_layers)
        self.extra_layers = (
            layer_utils.resolve_layers(list(cfg.extra_token_layers), model.n_layers)
            if cfg.extra_token_layers
            else []
        )
        self.token_layers = sorted({self.star_layer, *self.extra_layers})

    def extract(self, ex: Example, reasoning: str, predicted: str | None = None) -> ExtractionRecord:
        full_text = ex.prompt + reasoning
        chain_start = 0
        if self.cfg.chain_span == "completion":
            chain_start = len(self.model.tokenizer(ex.prompt, add_special_tokens=True).input_ids)

        hs = self.model.hidden_states(
            full_text,
            token_layers=self.token_layers if self.cfg.token_trajectory else [],
            want_layer_trajectory=self.cfg.layer_trajectory,
            chain_start=chain_start,
        )

        tok_trajs = {}
        if self.cfg.token_trajectory:
            for l, H in hs["token_hidden"].items():
                tt = build_token_trajectory({l: H}, star_layer=l, span=None, token_strings=hs["tokens"])
                tok_trajs[l] = tt

        layer_traj = None
        if self.cfg.layer_trajectory:
            layer_traj = build_layer_trajectory_from_means(hs["layer_means"])

        return ExtractionRecord(
            example_id=ex.id,
            prompt=ex.prompt,
            reasoning=reasoning,
            full_text=full_text,
            tokens=hs["tokens"],
            n_positions=hs["n_positions"],
            chain_start=hs["chain_start"],
            star_layer=self.star_layer,
            token_trajectories=tok_trajs,
            layer_trajectory=layer_traj,
            answer_key=ex.answer_key,
            predicted=predicted,
            meta={
                "extra_token_layers": self.extra_layers,
                "chain_span": self.cfg.chain_span,
                **ex.meta,
            },
        )
