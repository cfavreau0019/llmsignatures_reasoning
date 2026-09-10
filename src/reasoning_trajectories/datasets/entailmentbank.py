"""EntailmentBank loader + entailment-tree processing.

EntailmentBank (Dalvi et al., EMNLP 2021) ships **gold multi-step reasoning**:
each item is a hypothesis plus a tree of entailment steps that derives it from a
set of supporting facts. Unlike OpenBookQA there is nothing to generate -- the
proof *is* the reasoning chain -- so the natural pairing is
``reasoning.source: dataset_field``.

Getting the data
----------------
No reliable Hugging Face mirror (the popular one has a schema bug), so the
primary path is the official release:

    https://allenai.org/data/entailmentbank
    -> entailment_trees_emnlp2021_data_v3.zip

Unzip it and point ``dataset.params.local_dir`` at the folder that contains
``dataset/task_1/{train,dev,test}.jsonl`` (either the archive root or the
``dataset/`` folder works). Splits: ``train`` / ``dev`` (aka ``validation``) /
``test``. Tasks: ``1`` (gold leaves only), ``2`` (+25 distractors), ``3`` (corpus).

Alternatively pass ``dataset.params.data_files`` (a single ``.jsonl`` path or a
``datasets``-style split->path mapping) or ``dataset.params.hf_path``.

Record schema (v3)
------------------
Top level: ``id, context, question, answer, hypothesis, proof, full_text_proof,
depth_of_proof, length_of_proof, meta``.
``proof``: ``"sent2 & sent3 -> int1: <text>; int1 & sent1 -> hypothesis;"``.
``meta.triples``: ``{sentN: text}``; ``meta.intermediate_conclusions``:
``{intN: text}``; ``meta.distractors``: ``[sentN, ...]``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from ..reasoning.chain import ReasoningChain
from .base import Example

_SENT_SPLIT = re.compile(r"(sent\d+|int\d+):\s*")
_PROOF_PREFIX = re.compile(r"^\s*\$?proof\$?\s*=\s*", re.IGNORECASE)
_SENT_KEY = re.compile(r"^(?:sent|int)(\d+)$")

_PROMPT_STATEMENT = """Use the numbered facts to explain, step by step, why the statement is true.

Facts:
{facts}

Statement: {hypothesis}

Reasoning:"""

_PROMPT_QUESTION = """Use the numbered facts to answer the question. Reason step by step.

Facts:
{facts}

Question: {question}

Reasoning:"""


# --------------------------------------------------------------------------- #
# entailment-tree processing
# --------------------------------------------------------------------------- #
def parse_context(context: str) -> dict[str, str]:
    """``"sent1: a. sent2: b."`` -> ``{"sent1": "a.", "sent2": "b."}``."""
    parts = _SENT_SPLIT.split(context.strip())
    out: dict[str, str] = {}
    # parts = ['', key, text, key, text, ...]
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1].strip()
    return out


def build_id2text(record: dict) -> dict[str, str]:
    """Map every ``sentN`` / ``intN`` id (and ``hypothesis``) to its sentence."""
    meta = record.get("meta") or {}
    m: dict[str, str] = {}
    m.update(parse_context(record.get("context", "")))          # always available
    m.update(meta.get("triples") or {})                          # authoritative if present
    m.update(meta.get("intermediate_conclusions") or {})
    hyp = record.get("hypothesis") or meta.get("hypothesis") or meta.get("hypothesis_text")
    if hyp:
        m["hypothesis"] = hyp
    return m


def parse_proof_steps(proof: str) -> list[dict]:
    """Linear proof string -> ``[{premises: [...], conclusion_id, conclusion_text}]``."""
    proof = _PROOF_PREFIX.sub("", (proof or "").strip())
    steps: list[dict] = []
    for chunk in proof.split(";"):
        chunk = chunk.strip()
        if not chunk or "->" not in chunk:
            continue
        lhs, rhs = chunk.split("->", 1)
        premises = [p.strip() for p in lhs.split("&") if p.strip()]
        rhs = rhs.strip()
        if rhs.lower().startswith("hypothesis"):
            cid, ctext = "hypothesis", None
        elif ":" in rhs:
            cid, ctext = rhs.split(":", 1)
            cid, ctext = cid.strip(), ctext.strip()
        else:
            cid, ctext = rhs, None
        steps.append({"premises": premises, "conclusion_id": cid, "conclusion_text": ctext})
    return steps


def steps_from_proof(proof: str, id2text: dict[str, str]) -> list[dict]:
    """Resolve a linear proof string against an id->text map into id-annotated
    step dicts: ``{step, premises:[text], premise_ids:[id], conclusion, conclusion_id}``."""
    out: list[dict] = []
    for i, s in enumerate(parse_proof_steps(proof), 1):
        out.append({
            "step": i,
            "premise_ids": list(s["premises"]),
            "premises": [id2text.get(p, p) for p in s["premises"]],
            "conclusion_id": s["conclusion_id"],
            "conclusion": s["conclusion_text"] or id2text.get(s["conclusion_id"], s["conclusion_id"]),
        })
    return out


def resolve_steps(record: dict, *, tree_record: dict | None = None) -> list[dict]:
    """Parse a record's proof into resolved, id-annotated steps.

    ``tree_record`` supplies the proof + id->text mapping when it lives in a
    different record (task 3 carries no proof of its own, so the matching task-1
    record is passed here).
    """
    src = tree_record or record
    return steps_from_proof(src.get("proof", ""), build_id2text(src))


def format_steps(steps: list[dict], fmt: str = "natural") -> str:
    """Render resolved step dicts to reasoning text. Thin wrapper over
    :func:`reasoning_trajectories.reasoning.chain.render_steps`."""
    from ..reasoning.chain import Step, render_steps

    return render_steps([Step.from_dict(s) for s in steps], fmt)


def linearize_proof(
    record: dict, fmt: str = "natural", *, tree_record: dict | None = None
) -> tuple[str, list[dict]]:
    """Return ``(reasoning_text, resolved_steps)`` for a record's gold proof."""
    steps = resolve_steps(record, tree_record=tree_record)
    return format_steps(steps, fmt), steps


def facts_block(sent_map: dict[str, str], only: set[str] | None = None) -> str:
    def order(k: str) -> int:
        m = _SENT_KEY.match(k)
        return int(m.group(1)) if m else 1_000
    keys = [k for k in sent_map if k.startswith("sent") and (only is None or k in only)]
    return "\n".join(f"{order(k)}. {sent_map[k]}" for k in sorted(keys, key=order))


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
_SPLIT_ALIASES = {"validation": "dev", "valid": "dev", "val": "dev", "dev": "dev",
                  "train": "train", "test": "test"}

# conventional location of the unzipped v3 release, checked if nothing is configured
_DEFAULT_LOCAL_DIR = "data/raw/entailment_trees_emnlp2021_data_v3"


def _find_default_local() -> str | None:
    """Look for the unzipped release at the conventional path, walking up from
    the cwd (so it resolves whether you run from the repo root or a subdir)."""
    rel = Path(_DEFAULT_LOCAL_DIR)
    for base in (Path.cwd(), *list(Path.cwd().parents)[:5]):
        for cand in (base / rel, base / rel.name):
            if (cand / "dataset").is_dir():
                return str(cand)
    return None


class EntailmentBank:
    name = "entailmentbank"

    def __init__(
        self,
        task: int = 1,
        split: str = "validation",
        local_dir: str | Path | None = None,
        data_files: str | dict | None = None,
        hf_path: str | None = None,
        limit: int | None = None,
        shuffle_seed: int | None = 0,
        reasoning_format: str = "natural",
        use_question: bool = False,
    ):
        self.task = int(task)
        self.split = _SPLIT_ALIASES.get(split, split)
        self.reasoning_format = reasoning_format
        self.use_question = use_question
        self._gold_trees: dict[str, dict] = {}

        records = self._load(local_dir, data_files, hf_path)
        if shuffle_seed is not None:
            import random

            random.Random(shuffle_seed).shuffle(records)
        if limit is not None:
            records = records[:limit]
        self._records = records

    # ------------------------------------------------------------------ #
    def _load(self, local_dir, data_files, hf_path) -> list[dict]:
        if data_files is not None:
            files = data_files[self.split] if isinstance(data_files, dict) else data_files
            return _read_jsonl(files)
        if local_dir is None and data_files is None and hf_path is None:
            local_dir = _find_default_local()
        if local_dir is not None:
            records = _read_jsonl(self._resolve_local(local_dir, self.task))
            # task 3 has no proof of its own -> borrow the matching task-1 tree
            if any(not (r.get("proof") or "").strip() for r in records):
                try:
                    gold = _read_jsonl(self._resolve_local(local_dir, 1))
                    self._gold_trees = {str(r["id"]): r for r in gold}
                except FileNotFoundError:
                    pass
            return records
        if hf_path is not None:
            from datasets import load_dataset

            ds = load_dataset(hf_path, split={"dev": "validation"}.get(self.split, self.split))
            return [dict(r) for r in ds]
        raise ValueError(
            "EntailmentBank could not find its data. Either place the unzipped "
            f"release at '{_DEFAULT_LOCAL_DIR}/' (containing dataset/task_N/*.jsonl) "
            "or set dataset.params.local_dir / .data_files / .hf_path. "
            "Download entailment_trees_emnlp2021_data_v3.zip from "
            "https://allenai.org/data/entailmentbank"
        )

    def _resolve_local(self, local_dir: str | Path, task: int) -> Path:
        root = Path(local_dir)
        candidates = [
            root / "dataset" / f"task_{task}" / f"{self.split}.jsonl",
            root / f"task_{task}" / f"{self.split}.jsonl",
            root / f"{self.split}.jsonl",
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"No EntailmentBank task_{task}/{self.split}.jsonl under {root}. Tried: "
            + ", ".join(str(c) for c in candidates)
        )

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Example]:
        for rec in self._records:
            meta_in = rec.get("meta") or {}
            id2text = build_id2text(rec)
            sent_map = {k: v for k, v in id2text.items() if k.startswith("sent")}
            hypothesis = id2text.get("hypothesis", "")
            question = rec.get("question") or meta_in.get("question_text") or ""
            answer = rec.get("answer") or meta_in.get("answer_text") or ""

            tree_rec = rec
            if not (rec.get("proof") or "").strip():
                tree_rec = self._gold_trees.get(str(rec.get("id")), rec)
            chain = ReasoningChain.from_entailmentbank(rec, tree_record=tree_rec, fmt=self.reasoning_format)
            steps = chain.to_dicts()
            reasoning_text = chain.text
            facts = facts_block(sent_map)

            if self.use_question and question:
                prompt = _PROMPT_QUESTION.format(facts=facts, question=question.strip())
            else:
                prompt = _PROMPT_STATEMENT.format(facts=facts, hypothesis=hypothesis.strip() or answer)

            yield Example(
                id=str(rec.get("id", f"eb_{len(steps)}")),
                question=question or hypothesis,
                choices=[answer] if answer else [],
                choice_labels=["A"] if answer else [],
                answer_key="A" if answer else None,
                prompt=prompt,
                context=facts,
                meta={
                    "dataset": self.name,
                    "task": self.task,
                    "split": self.split,
                    "hypothesis": hypothesis,
                    "answer_text": answer,
                    "reasoning": reasoning_text,          # consumed by dataset_field
                    "reasoning_format": self.reasoning_format,
                    "entailment_steps": steps,
                    "proof_raw": _PROOF_PREFIX.sub("", (tree_rec.get("proof") or "").strip()),
                    "proof_borrowed_from_task1": tree_rec is not rec,
                    "depth_of_proof": rec.get("depth_of_proof") or tree_rec.get("depth_of_proof"),
                    "length_of_proof": (rec.get("length_of_proof")
                                        or tree_rec.get("length_of_proof") or len(steps)),
                    "n_distractors": len(meta_in.get("distractors") or []),
                },
            )


def _read_jsonl(path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
