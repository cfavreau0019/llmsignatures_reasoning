"""OpenBookQA loader.

HF dataset ``allenai/openbookqa`` (configs ``main`` and ``additional``).
Each row: ``id``, ``question_stem``, ``choices={text[], label[]}``, ``answerKey``.
The ``additional`` config also carries ``fact1`` -- the gold science fact -- which
can be prepended to the prompt with ``include_fact=True``.
"""
from __future__ import annotations

from typing import Iterator

from .base import Example

_PROMPT = """Answer the multiple-choice question below. Reason step by step, then \
end with a line of exactly the form "Answer: <letter>".
{context}Question: {stem}
{options}

Reasoning:"""


def _format_options(labels: list[str], texts: list[str]) -> str:
    return "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))


def format_prompt(stem: str, labels: list[str], texts: list[str], fact: str | None = None) -> str:
    context = f"Fact: {fact}\n" if fact else ""
    return _PROMPT.format(context=context, stem=stem.strip(), options=_format_options(labels, texts))


class OpenBookQA:
    name = "openbookqa"

    def __init__(
        self,
        hf_path: str = "allenai/openbookqa",
        hf_config: str = "main",
        split: str = "validation",
        limit: int | None = None,
        include_fact: bool = False,
        shuffle_seed: int | None = 0,
    ):
        from datasets import load_dataset

        ds = load_dataset(hf_path, hf_config, split=split)
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        self._ds = ds
        self.include_fact = include_fact
        self.hf_config = hf_config

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self) -> Iterator[Example]:
        for row in self._ds:
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            fact = row.get("fact1") if self.include_fact else None
            stem = row["question_stem"]
            yield Example(
                id=str(row["id"]),
                question=stem,
                choices=texts,
                choice_labels=labels,
                answer_key=row.get("answerKey") or None,
                prompt=format_prompt(stem, labels, texts, fact),
                context=fact,
                meta={"dataset": self.name, "hf_config": self.hf_config},
            )
