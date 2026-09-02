"""Container for one example's extraction result (no heavy deps)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..trajectories import LayerTrajectory, TokenTrajectory


@dataclass
class ExtractionRecord:
    example_id: str
    prompt: str
    reasoning: str
    full_text: str
    tokens: list[str]
    n_positions: int
    chain_start: int
    star_layer: int
    token_trajectories: dict[int, TokenTrajectory]      # layer -> token trajectory
    layer_trajectory: LayerTrajectory | None
    answer_key: str | None = None
    predicted: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_correct(self) -> bool | None:
        if self.answer_key is None or self.predicted is None:
            return None
        return self.predicted == self.answer_key
