"""``ReasoningChain`` -- a single reasoning chain as a first-class value.

A chain bundles what used to be spread across ``meta["reasoning"]`` (text),
``meta["entailment_steps"]`` (structure) and the loose helpers in
``datasets/entailmentbank.py`` / ``datasets/perturbations.py``. It can be built
from one source (a raw EntailmentBank record, a list of steps, a proof string,
or opaque generated text), inspected, re-formatted, and corrupted -- all without
touching a dataset or the pair machinery.

    ch = ReasoningChain.from_entailmentbank(record)      # or .from_steps(...)
    ch.text                       # rendered in ch.fmt
    ch.premises, ch.hypothesis    # components
    ch.as_format("structured")    # -> new chain, different render
    bad = ch.corrupt("step_shuffle", seed=0)             # -> new chain
    bad.meta["corruption"]        # {"kind": "step_shuffle", "permutation": [...]}
    ch.text                       # original untouched

Every transformation returns a **new** ``ReasoningChain``; the change record for
a corruption lands in ``.meta["corruption"]``.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

# every corruption kind ``.corrupt()`` understands
CORRUPTIONS = (
    "step_shuffle",
    "premise_substitution",             # same-tree premise swap (also the short-chain fallback)
    "premise_substitution_other_tree",
    "conclusion_substitution",
)


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One entailment step: some premises jointly entail a conclusion."""
    premises: tuple[str, ...]
    conclusion: str
    premise_ids: tuple[str, ...] = ()          # sentN / intN / "SUB"  (may be empty)
    conclusion_id: str = ""                    # "intN" | "hypothesis" | "SUB" | ""

    @property
    def is_final(self) -> bool:
        return self.conclusion_id == "hypothesis"

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            premises=tuple(d["premises"]),
            conclusion=d["conclusion"],
            premise_ids=tuple(d.get("premise_ids") or ()),
            conclusion_id=d.get("conclusion_id") or "",
        )


# --------------------------------------------------------------------------- #
def render_steps(steps: Sequence[Step], fmt: str = "natural") -> str:
    """Render steps to reasoning text. Steps are numbered by position, so a
    reordered list renumbers automatically. Premises are bracketed because
    EntailmentBank sentences contain their own ``;`` and ``,``."""
    if not steps:
        return ""
    if fmt == "raw":
        return "; ".join(" & ".join(s.premises) + " -> " + s.conclusion for s in steps) + ";"
    lines = []
    for i, s in enumerate(steps, 1):
        if fmt == "structured":
            lines.append(f"Step {i}: " + " + ".join(f"[{p}]" for p in s.premises)
                         + f"  =>  {s.conclusion}")
        else:  # natural (default)
            prem = ", and ".join(f"[{p}]" for p in s.premises)
            lines.append(f"{i}. Since {prem}, it follows that {s.conclusion}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
@dataclass
class ReasoningChain:
    steps: list[Step]
    fmt: str = "natural"                       # natural | structured | raw
    source_id: str | None = None
    meta: dict = field(default_factory=dict)
    raw_text: str | None = None               # set for opaque (from_text) chains

    # -- text --------------------------------------------------------------- #
    @property
    def text(self) -> str:
        if self.raw_text is not None:
            return self.raw_text
        return render_steps(self.steps, self.fmt)

    @property
    def is_structured(self) -> bool:
        return self.raw_text is None and bool(self.steps)

    def as_format(self, fmt: str) -> "ReasoningChain":
        return replace(self, fmt=fmt, meta=dict(self.meta))

    # -- components ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def premises(self) -> list[str]:
        return [p for s in self.steps for p in s.premises]

    @property
    def conclusions(self) -> list[str]:
        return [s.conclusion for s in self.steps]

    @property
    def hypothesis(self) -> str | None:
        for s in reversed(self.steps):
            if s.is_final:
                return s.conclusion
        return self.steps[-1].conclusion if self.steps else self.raw_text

    @property
    def leaves(self) -> list[str]:
        """Distinct premise sentences that are tree leaves (id ``sentN``)."""
        seen: list[str] = []
        for s in self.steps:
            for p, pid in zip(s.premises, list(s.premise_ids) + [""] * len(s.premises)):
                if pid.startswith("sent") and p not in seen:
                    seen.append(p)
        return seen

    @property
    def intermediates(self) -> list[str]:
        return [s.conclusion for s in self.steps if not s.is_final]

    def to_dicts(self) -> list[dict]:
        """The ``meta['entailment_steps']`` schema (storage / back-compat)."""
        return [
            {"step": i, "premise_ids": list(s.premise_ids), "premises": list(s.premises),
             "conclusion_id": s.conclusion_id, "conclusion": s.conclusion}
            for i, s in enumerate(self.steps, 1)
        ]

    # -- construction ---------------------------------------------------- #
    @classmethod
    def from_steps(cls, steps: Iterable, *, fmt: str = "natural",
                   source_id: str | None = None, meta: dict | None = None) -> "ReasoningChain":
        out = [s if isinstance(s, Step) else Step.from_dict(s) for s in steps]
        return cls(steps=out, fmt=fmt, source_id=source_id, meta=dict(meta or {}))

    @classmethod
    def from_entailmentbank(cls, record: dict, *, tree_record: dict | None = None,
                            fmt: str = "natural") -> "ReasoningChain":
        from ..datasets.entailmentbank import resolve_steps

        sid = str(record["id"]) if record.get("id") else None
        return cls.from_steps(resolve_steps(record, tree_record=tree_record), fmt=fmt, source_id=sid)

    @classmethod
    def from_proof(cls, proof: str, id2text: dict[str, str], *, fmt: str = "natural",
                   source_id: str | None = None) -> "ReasoningChain":
        from ..datasets.entailmentbank import steps_from_proof

        return cls.from_steps(steps_from_proof(proof, id2text), fmt=fmt, source_id=source_id)

    @classmethod
    def from_text(cls, text: str, *, source_id: str | None = None) -> "ReasoningChain":
        """An opaque chain (e.g. model-generated prose): no step structure, but
        ``.text`` and the transformation methods still work (corruptions no-op)."""
        return cls(steps=[], fmt="text", source_id=source_id, raw_text=text or "")

    @classmethod
    def from_example(cls, ex, *, fmt: str | None = None) -> "ReasoningChain":
        steps = ex.meta.get("entailment_steps")
        keep = {k: ex.meta[k] for k in ("condition", "pair_id", "corruption") if k in ex.meta}
        if steps:
            return cls.from_steps(steps, source_id=getattr(ex, "id", None),
                                  fmt=fmt or ex.meta.get("reasoning_format", "natural"), meta=keep)
        chain = cls.from_text(ex.meta.get("reasoning") or ex.context or "",
                              source_id=getattr(ex, "id", None))
        chain.meta.update(keep)
        return chain

    # -- transformation (all return a new chain) ------------------------ #
    def _corrupted(self, new_steps: Sequence[Step], info: dict) -> "ReasoningChain":
        meta = dict(self.meta)
        meta["corruption"] = info
        return ReasoningChain(steps=list(new_steps), fmt=self.fmt, source_id=self.source_id,
                              meta=meta, raw_text=self.raw_text)

    def shuffled(self, *, seed: int = 0, rng: random.Random | None = None) -> "ReasoningChain":
        rng = rng if rng is not None else _rng_for(seed, self.source_id)
        if len(self.steps) < 2:
            return self._corrupted(self.steps, {"kind": "step_shuffle", "skipped": "fewer than 2 steps"})
        perm = _non_identity_permutation(len(self.steps), rng)
        return self._corrupted([self.steps[i] for i in perm],
                               {"kind": "step_shuffle", "permutation": perm})

    def substitute_premise(self, *, other_tree: bool = True, donors: "DonorPool | None" = None,
                           seed: int = 0, rng: random.Random | None = None) -> "ReasoningChain":
        rng = rng if rng is not None else _rng_for(seed, self.source_id)
        kind = "premise_substitution_other_tree" if other_tree else "premise_substitution"
        if not self.steps:
            return self._corrupted(self.steps, {"kind": kind, "skipped": "no steps"})
        steps = list(self.steps)
        si = rng.randrange(len(steps))
        pi = rng.randrange(len(steps[si].premises))
        old = steps[si].premises[pi]
        own = {p for s in steps for p in s.premises} | {s.conclusion for s in steps}
        if other_tree:
            if donors is None:
                raise ValueError("premise_substitution_other_tree needs donors=DonorPool(...)")
            new = donors.premise(rng, exclude_texts=own, exclude_src=self.source_id)
        else:
            pool = sorted(own - set(steps[si].premises))
            new = rng.choice(pool) if pool else (
                donors.premise(rng, own, self.source_id) if donors is not None else "[unrelated fact]")
        prem = list(steps[si].premises); prem[pi] = new
        pids = list(steps[si].premise_ids)
        if pids:
            pids[pi] = "SUB"
        steps[si] = replace(steps[si], premises=tuple(prem), premise_ids=tuple(pids))
        return self._corrupted(steps, {"kind": kind, "step": si + 1, "premise_index": pi,
                                       "old": old, "new": new})

    def substitute_conclusion(self, *, donors: "DonorPool", seed: int = 0,
                              rng: random.Random | None = None) -> "ReasoningChain":
        rng = rng if rng is not None else _rng_for(seed, self.source_id)
        if not self.steps:
            return self._corrupted(self.steps, {"kind": "conclusion_substitution", "skipped": "no steps"})
        steps = list(self.steps)
        cand = [i for i, s in enumerate(steps) if s.conclusion_id != "hypothesis"] or list(range(len(steps)))
        si = rng.choice(cand)
        old, old_id = steps[si].conclusion, steps[si].conclusion_id
        own = {p for s in steps for p in s.premises} | {s.conclusion for s in steps}
        new = donors.conclusion(rng, exclude_texts=own, exclude_src=self.source_id)
        steps[si] = replace(steps[si], conclusion=new, conclusion_id="SUB")
        propagated = 0
        if old_id and old_id != "hypothesis":
            for j in range(si + 1, len(steps)):
                pids, prem, hit = list(steps[j].premise_ids), list(steps[j].premises), False
                for k, pid in enumerate(pids):
                    if pid == old_id:
                        prem[k], pids[k], hit = new, "SUB", True
                        propagated += 1
                if hit:
                    steps[j] = replace(steps[j], premises=tuple(prem), premise_ids=tuple(pids))
        return self._corrupted(steps, {"kind": "conclusion_substitution", "step": si + 1,
                                       "old": old, "new": new, "propagated_to": propagated})

    def corrupt(self, kind: str, *, seed: int = 0, donors: "DonorPool | None" = None,
                rng: random.Random | None = None, min_steps: int = 3) -> "ReasoningChain":
        """Dispatch to one corruption. Returns a new chain; ``.meta['corruption']``
        holds the change record. ``donors`` is required only for the
        ``*_other_tree`` and ``conclusion_substitution`` kinds."""
        if kind not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {kind!r}; choose from {list(CORRUPTIONS)}")
        rng = rng if rng is not None else _rng_for(seed, self.source_id)

        if self.raw_text is not None:                       # opaque chain -> nothing to break
            return self._corrupted(self.steps, {"kind": kind, "skipped": "opaque text chain"})

        if kind == "step_shuffle":
            if len(self.steps) >= max(3, min_steps):
                return self.shuffled(rng=rng)
            kind = "premise_substitution"                   # escalate short chains

        if kind in ("premise_substitution", "premise_substitution_other_tree"):
            return self.substitute_premise(other_tree=kind.endswith("other_tree"), donors=donors, rng=rng)
        return self.substitute_conclusion(donors=donors, rng=rng)

    def __repr__(self) -> str:
        n = len(self.steps) if self.is_structured else f"text[{len(self.text)}]"
        return f"ReasoningChain(source_id={self.source_id!r}, {n}, fmt={self.fmt!r})"


# --------------------------------------------------------------------------- #
class DonorPool:
    """Premise / conclusion sentences harvested from many chains, tagged with
    their source id so the current chain can be excluded. Accepts an iterable of
    :class:`ReasoningChain` **or** ``Example``-like objects
    (``.meta['entailment_steps']`` + ``.id``)."""

    def __init__(self, items: Iterable):
        self._prem: list[tuple[str, str | None]] = []
        self._concl: list[tuple[str, str | None]] = []
        for item in items:
            if isinstance(item, ReasoningChain):
                steps, sid = item.steps, item.source_id
            else:
                steps = [Step.from_dict(d) for d in (item.meta.get("entailment_steps") or [])]
                sid = getattr(item, "id", None)
            for s in steps:
                for p in s.premises:
                    self._prem.append((p, sid))
                self._concl.append((s.conclusion, sid))

    @staticmethod
    def _pick(table, rng, exclude_texts, exclude_src) -> str:
        pool = sorted({t for t, src in table if src != exclude_src and t not in exclude_texts})
        if not pool:
            pool = sorted({t for t, _ in table if t not in exclude_texts})
        return rng.choice(pool) if pool else "[unrelated fact]"

    def premise(self, rng, exclude_texts, exclude_src) -> str:
        return self._pick(self._prem, rng, exclude_texts, exclude_src)

    def conclusion(self, rng, exclude_texts, exclude_src) -> str:
        return self._pick(self._concl, rng, exclude_texts, exclude_src)


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


def _rng_for(seed: int, source_id: str | None) -> random.Random:
    """Deterministic per-chain RNG (not Python ``hash`` -- reproducible across
    processes)."""
    h = hashlib.blake2b(f"{seed}:{source_id or ''}".encode(), digest_size=8)
    return random.Random(int.from_bytes(h.digest(), "big"))
