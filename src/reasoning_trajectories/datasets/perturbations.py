"""Paired valid/corrupted reasoning chains for the trajectory-geometry study.

The corruption mechanics now live on :class:`ReasoningChain`
(``chain.corrupt(kind, seed=, donors=)``); this module is just the **dataset
wrapper** that turns a base dataset into matched valid/corrupted pairs.

Design (matches the study spec)
-------------------------------
* **Valid set** -- gold trees with at least ``min_steps`` (default 3) materialized
  entailment steps. Each is emitted unchanged as the ``valid`` member of a pair.
* Every valid chain is paired with **one corruption**, its type assigned
  **round-robin** over ``cycle`` (default: ``step_shuffle`` /
  ``premise_substitution_other_tree`` / ``conclusion_substitution``), so the
  three families are balanced across the dataset. Chains too short to shuffle
  escalate to a same-tree ``premise_substitution``.
* All randomness is seeded per example from ``seed`` (default 0), via
  ``ReasoningChain.corrupt(seed=...)`` -> reproducible across processes.

Works on any dataset whose ``Example`` objects carry ``meta["entailment_steps"]``.
"""
from __future__ import annotations

from dataclasses import replace

from ..reasoning.chain import CORRUPTIONS, DonorPool, ReasoningChain
from .base import Example

CYCLE_DEFAULT = ("step_shuffle", "premise_substitution_other_tree", "conclusion_substitution")

__all__ = ["CYCLE_DEFAULT", "CORRUPTIONS", "DonorPool", "PairedCorruption", "apply_corruption"]


def apply_corruption(steps: list[dict], kind: str, *, rng, donors: DonorPool,
                     src_id: str, min_steps: int = 3) -> tuple[list[dict], dict]:
    """Back-compat shim. Prefer ``ReasoningChain.from_steps(steps).corrupt(...)``.

    Returns ``(corrupted_step_dicts, info)``.
    """
    out = ReasoningChain.from_steps(steps, source_id=src_id).corrupt(
        kind, rng=rng, donors=donors, min_steps=min_steps
    )
    return out.to_dicts(), out.meta["corruption"]


# --------------------------------------------------------------------------- #
class PairedCorruption:
    """Wrap a dataset; emit ``(valid, corrupted)`` chain pairs.

    Each yielded :class:`Example` gains ``meta["condition"]``
    (``"valid"``/``"corrupted"``), ``meta["pair_id"]`` (shared by the two
    members) and, for corrupted ones, ``meta["corruption"]``. Corrupted ids are
    ``"<pair_id>__<kind>"``.
    """

    name = "paired-corruption"

    def __init__(self, base, *, min_steps: int = 3, seed: int = 0,
                 cycle=CYCLE_DEFAULT, emit: str = "both", reasoning_format: str = "natural"):
        if emit not in ("both", "valid", "corrupted"):
            raise ValueError("emit must be 'both', 'valid' or 'corrupted'")
        self.min_steps = int(min_steps)
        self.seed = int(seed)
        self.cycle = tuple(cycle) or CYCLE_DEFAULT
        self.emit = emit
        self.fmt = reasoning_format
        self.base_name = getattr(base, "name", "dataset")

        self._base = list(base)
        self._chains = {
            ex.id: ReasoningChain.from_steps(ex.meta.get("entailment_steps") or [],
                                             fmt=self.fmt, source_id=ex.id)
            for ex in self._base
        }
        self._valid = [ex for ex in self._base
                       if len(ex.meta.get("entailment_steps") or []) >= self.min_steps]
        self._donors = DonorPool(self._chains.values())

    def __len__(self) -> int:
        return len(self._valid) * (2 if self.emit == "both" else 1)

    def assignments(self) -> list[tuple[str, str]]:
        """(example_id, corruption_kind) in emission order -- handy for inspection."""
        return [(ex.id, self.cycle[i % len(self.cycle)]) for i, ex in enumerate(self._valid)]

    def __iter__(self):
        for i, ex in enumerate(self._valid):
            kind = self.cycle[i % len(self.cycle)]
            chain = self._chains[ex.id]
            corrupted = chain.corrupt(kind, seed=self.seed, donors=self._donors,
                                      min_steps=self.min_steps)
            if self.emit in ("both", "valid"):
                yield _variant(ex, "valid", ex.id, chain)
            if self.emit in ("both", "corrupted"):
                yield _variant(ex, "corrupted", ex.id, corrupted)


def _variant(ex: Example, condition: str, pair_id: str, chain: ReasoningChain) -> Example:
    corruption = chain.meta.get("corruption")
    meta = dict(ex.meta)
    meta["entailment_steps"] = chain.to_dicts()
    meta["reasoning"] = chain.text
    meta["condition"] = condition
    meta["pair_id"] = pair_id
    meta["corruption"] = corruption
    new_id = pair_id if condition == "valid" else f"{pair_id}__{(corruption or {}).get('kind', 'corrupt')}"
    return replace(ex, id=new_id, meta=meta)
