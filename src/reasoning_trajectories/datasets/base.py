"""Dataset-agnostic reasoning example schema.

Adding a new multiple-choice / short-answer dataset means writing a loader that
yields :class:`Example` objects; the rest of the pipeline is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Example:
    id: str
    question: str
    choices: list[str]                 # answer options, in order
    choice_labels: list[str]           # e.g. ["A", "B", "C", "D"]
    answer_key: str | None             # gold label, e.g. "C" (None if unknown)
    prompt: str                        # fully formatted model input up to "Reasoning:"
    context: str | None = None         # optional supporting fact / passage
    meta: dict = field(default_factory=dict)

    @property
    def answer_index(self) -> int | None:
        if self.answer_key is None:
            return None
        try:
            return self.choice_labels.index(self.answer_key)
        except ValueError:
            return None


class ReasoningDataset(Protocol):
    name: str

    def __iter__(self) -> Iterator[Example]: ...

    def __len__(self) -> int: ...
