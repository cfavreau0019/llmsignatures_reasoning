"""Paired reasoning-chain corruptions for the trajectory-geometry study.

Design (matches the study spec)
-------------------------------
* **Valid set** -- gold trees with at least ``min_steps`` (default 3) materialized
  entailment steps. Each is emitted unchanged as the ``valid`` member of a pair.
* Every valid chain is paired with **one corruption**, its type assigned
  **round-robin** over ``cycle`` (default below), so the three corruption
  families are balanced across the dataset:

  1. ``step_shuffle`` -- reorder the steps with a *guaranteed non-identity*
     permutation. Every step stays locally sound and every sentence stays intact;
     what breaks is the global derivation order. A chain with fewer than
     ``min_steps`` steps (rare, given the filter) *escalates* to
     ``premise_substitution``.
  2. ``premise_substitution_other_tree`` -- replace one premise of one step with
     a sentence drawn from a *different* tree. That step becomes a non-sequitur
     and a topically foreign sentence enters the chain.
  3. ``conclusion_substitution`` -- replace one step's derived conclusion with a
     sentence from another tree. The step's premises no longer support it and any
     later step that consumed that intermediate now builds on a wrong fact.

  (``premise_substitution`` -- same-tree premise swap -- exists only as the
  short-chain escalation target.)

All randomness (the permutation, which step/premise, which donor sentence) is
seeded per example from ``seed`` so a run is fully reproducible; ``seed=0`` by
default.

The wrapper is dataset-agnostic: it works on any dataset whose ``Example``
objects carry ``meta["entailment_steps"]`` in the shape produced by
:func:`reasoning_trajectories.datasets.entailmentbank.resolve_steps`.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .base import Example
from .entailmentbank import format_steps

CYCLE_DEFAULT = ("step_shuffle", "premise_substitution_other_tree", "conclusion_substitution")
_ALL_KINDS = set(CYCLE_DEFAULT) | {"premise_substitution"}


# --------------------------------------------------------------------------- #
# donor material
# --------------------------------------------------------------------------- #
class DonorPool:
    """Premise / conclusion sentences harvested from a set of examples, tagged
    with their source id so the current tree can be excluded."""

    def __init__(self, examples):
        self._prem: list[tuple[str, str]] = []   # (text, src_id)
        self._concl: list[tuple[str, str]] = []
        for ex in examples:
            for s in ex.meta.get("entailment_steps") or []:
                for p in s["premises"]:
                    self._prem.append((p, ex.id))
                self._concl.append((s["conclusion"], ex.id))

    def _pick(self, table, rng, exclude_texts, exclude_src) -> str:
        pool = sorted({t for t, src in table if src != exclude_src and t not in exclude_texts})
        if not pool:  # degenerate corpus (e.g. 1 tree): fall back to anything unseen
            pool = sorted({t for t, _ in table if t not in exclude_texts})
        if not pool:
            return "[unrelated fact]"
        return rng.choice(pool)

    def premise(self, rng, exclude_texts, exclude_src) -> str:
        return self._pick(self._prem, rng, exclude_texts, exclude_src)

    def conclusion(self, rng, exclude_texts, exclude_src) -> str:
        return self._pick(self._concl, rng, exclude_texts, exclude_src)


# --------------------------------------------------------------------------- #
# the corruptions
# --------------------------------------------------------------------------- #
def _non_identity_permutation(k: int, rng: random.Random) -> list[int]:
    idx = list(range(k))
    for _ in range(100):
        rng.shuffle(idx)
        if idx != list(range(k)):
            return list(idx)
    idx = list(range(k))                       # k >= 2 guarantees this differs
    idx[0], idx[1] = idx[1], idx[0]
    return idx


def _copy_steps(steps: list[dict]) -> list[dict]:
    return [dict(s, premises=list(s["premises"]), premise_ids=list(s.get("premise_ids", [])))
            for s in steps]


def apply_corruption(steps: list[dict], kind: str, *, rng: random.Random,
                     donors: DonorPool, src_id: str, min_steps: int = 3) -> tuple[list[dict], dict]:
    """Return ``(corrupted_steps, info)``. ``info`` records exactly what changed."""
    if kind not in _ALL_KINDS:
        raise ValueError(f"unknown corruption {kind!r}; choose from {sorted(_ALL_KINDS)}")
    steps = _copy_steps(steps)

    if kind == "step_shuffle":
        if len(steps) < max(3, min_steps):
            return apply_corruption(steps, "premise_substitution", rng=rng, donors=donors,
                                    src_id=src_id, min_steps=min_steps)
        perm = _non_identity_permutation(len(steps), rng)
        return [steps[i] for i in perm], {"kind": "step_shuffle", "permutation": perm}

    own_texts = {p for s in steps for p in s["premises"]} | {s["conclusion"] for s in steps}

    if kind in ("premise_substitution", "premise_substitution_other_tree"):
        si = rng.randrange(len(steps))
        pi = rng.randrange(len(steps[si]["premises"]))
        old = steps[si]["premises"][pi]
        if kind == "premise_substitution_other_tree":
            new = donors.premise(rng, exclude_texts=own_texts, exclude_src=src_id)
        else:  # same-tree: a sentence from this tree not already a premise of this step
            pool = sorted(own_texts - set(steps[si]["premises"]))
            new = rng.choice(pool) if pool else donors.premise(rng, own_texts, src_id)
        steps[si]["premises"][pi] = new
        if steps[si].get("premise_ids"):
            steps[si]["premise_ids"][pi] = "SUB"
        return steps, {"kind": kind, "step": si + 1, "premise_index": pi, "old": old, "new": new}

    # conclusion_substitution -- prefer an intermediate node so the hypothesis stays.
    # The wrong conclusion also propagates: any later step that consumed this
    # intermediate as a premise now carries the wrong fact.
    cand = [i for i, s in enumerate(steps) if s.get("conclusion_id") != "hypothesis"] \
        or list(range(len(steps)))
    si = rng.choice(cand)
    old = steps[si]["conclusion"]
    old_id = steps[si].get("conclusion_id")
    new = donors.conclusion(rng, exclude_texts=own_texts, exclude_src=src_id)
    steps[si]["conclusion"] = new
    steps[si]["conclusion_id"] = "SUB"
    propagated = 0
    if old_id not in (None, "hypothesis"):
        for later in steps[si + 1:]:
            for j, pid in enumerate(later.get("premise_ids", [])):
                if pid == old_id:
                    later["premises"][j] = new
                    later["premise_ids"][j] = "SUB"
                    propagated += 1
    return steps, {"kind": "conclusion_substitution", "step": si + 1, "old": old,
                   "new": new, "propagated_to": propagated}


def _seed_for(seed: int, example_id: str) -> int:
    h = hashlib.blake2b(f"{seed}:{example_id}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


# --------------------------------------------------------------------------- #
# the dataset wrapper
# --------------------------------------------------------------------------- #
class PairedCorruption:
    """Wrap a dataset; emit ``(valid, corrupted)`` chain pairs.

    Each yielded :class:`Example` gains
    ``meta["condition"]`` (``"valid"`` / ``"corrupted"``),
    ``meta["pair_id"]`` (shared by the two members), and, for corrupted ones,
    ``meta["corruption"]`` (the change record). Corrupted ids are
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
        self._valid = [ex for ex in self._base
                       if len(ex.meta.get("entailment_steps") or []) >= self.min_steps]
        self._donors = DonorPool(self._base)

    def __len__(self) -> int:
        per = 2 if self.emit == "both" else 1
        return len(self._valid) * per

    def assignments(self) -> list[tuple[str, str]]:
        """(example_id, corruption_kind) in emission order -- handy for inspection."""
        return [(ex.id, self.cycle[i % len(self.cycle)]) for i, ex in enumerate(self._valid)]

    def __iter__(self):
        for i, ex in enumerate(self._valid):
            kind = self.cycle[i % len(self.cycle)]
            rng = random.Random(_seed_for(self.seed, ex.id))
            steps = ex.meta["entailment_steps"]
            corrupt_steps, info = apply_corruption(
                steps, kind, rng=rng, donors=self._donors, src_id=ex.id, min_steps=self.min_steps
            )
            if self.emit in ("both", "valid"):
                yield _variant(ex, "valid", ex.id, steps, self.fmt, None)
            if self.emit in ("both", "corrupted"):
                yield _variant(ex, "corrupted", ex.id, corrupt_steps, self.fmt, info)


def _variant(ex: Example, condition: str, pair_id: str, steps: list[dict],
             fmt: str, corruption: dict | None) -> Example:
    meta = dict(ex.meta)
    meta["entailment_steps"] = steps
    meta["reasoning"] = format_steps(steps, fmt)
    meta["condition"] = condition
    meta["pair_id"] = pair_id
    meta["corruption"] = corruption
    new_id = pair_id if condition == "valid" else f"{pair_id}__{(corruption or {}).get('kind', 'corrupt')}"
    return replace(ex, id=new_id, meta=meta)
