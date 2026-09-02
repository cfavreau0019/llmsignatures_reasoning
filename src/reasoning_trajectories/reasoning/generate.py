"""Obtain the reasoning-chain text for an example and score its final answer."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..datasets.base import Example

_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?\s*([A-Ha-h1-8])\s*\)?", re.IGNORECASE)


def truncate_after_answer(text: str) -> str:
    """Keep everything up to and including the first 'Answer: X' line."""
    m = _ANSWER_RE.search(text)
    if not m:
        return text.strip()
    end = text.find("\n", m.end())
    return (text if end == -1 else text[:end]).strip()


def parse_answer(text: str, choice_labels: list[str]) -> str | None:
    """Return the predicted label, mapping numeric answers to positional labels."""
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    tok = m.group(1).upper()
    if tok in choice_labels:
        return tok
    if tok.isdigit():
        i = int(tok) - 1
        if 0 <= i < len(choice_labels):
            return choice_labels[i]
    return None


class ReasoningProvider:
    """Produces the CoT text for an :class:`Example` per the config's `source`."""

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        self._precomputed: dict[str, str] = {}
        if cfg.source == "precomputed":
            if not cfg.precomputed_path:
                raise ValueError("reasoning.source='precomputed' needs reasoning.precomputed_path")
            for line in Path(cfg.precomputed_path).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._precomputed[str(row["id"])] = row["reasoning"]

    def reasoning_for(self, ex: Example) -> str:
        cfg = self.cfg
        if cfg.source == "precomputed":
            text = self._precomputed[ex.id]
        elif cfg.source == "dataset_field":
            text = ex.meta.get("reasoning") or ex.context or ""
        elif cfg.source == "generate":
            if self.model is None:
                raise RuntimeError("reasoning.source='generate' requires a model")
            text = self.model.generate_reasoning(
                ex.prompt, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature
            )
        else:
            raise ValueError(f"Unknown reasoning.source {cfg.source!r}")

        if cfg.stop_on_answer:
            text = truncate_after_answer(text)
        return text
