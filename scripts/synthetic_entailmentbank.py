# /// script
# dependencies = [
#   "data-designer",
#   "pydantic",
#   "python-dotenv",
# ]
# ///
"""Synthetic EntailmentBank-style entailment trees, generated with NVIDIA Data Designer.

Design
------
LLMs are bad at hitting a target tree *shape*, so the shape is sampled, not prompted:

1. ``skeleton``   -- a custom column samples a proof graph (steps, fan-in, linear vs.
                     branched, occasional shared leaf) skewed toward 3+ steps, seeded per row.
2. ``tree``       -- an LLM writes the *sentences* for that exact skeleton (question, answer,
                     hypothesis, leaf facts, intermediate conclusions) as structured output.
3. ``record`` etc -- a deterministic custom column assembles the raw EntailmentBank task_1
                     record (``proof``, ``lisp_proof``, ``polish_proof``, ``full_text_proof``,
                     ``context``, ``meta`` ...), so every encoding agrees by construction, and
                     checks structural invariants.
4. ``judge``      -- a second model scores step validity, factual correctness, answer
                     consistency and non-triviality (1-5). Scores are kept as columns.

Task 2 / task 3 variants need cross-row information (distractors drawn from *other* records),
so they are built afterwards by ``scripts/export_synthetic_entailmentbank.py``, which imports
the pure helpers defined here.

Usage (from the repo root, with ``pip install data-designer`` in ``.venv``)::

    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py validate
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py preview -n 10
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py create -n 1400 --dataset-name entailmentbank_synth_v1

The driver calls Data Designer's Python API because the ``data-designer preview|create`` CLI commands
crash on Typer >= 0.2x (``click_type``). ``data-designer validate scripts/synthetic_entailmentbank.py``
still works (on Windows set ``PYTHONUTF8=1`` or it dies printing emoji), and this file exposes the usual
``load_config_builder()``.

Environment (loaded from the nearest ``.env``; real env vars win):
    OPENAI_API_KEY       read by Data Designer's built-in ``openai`` provider
    OPENAI_MODEL         model that writes the trees
    OPENAI_JUDGE_MODEL   model that judges them (falls back to OPENAI_MODEL)
A leading ``openai/`` in a model name is stripped: OpenAI's own API uses bare ids.

The pure helpers (``build_skeleton``, ``assemble_task1``, ``check_task1_record`` ...) need no
Data Designer install, so they are importable from tests and the export script.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

try:  # the pure helpers below stay importable without Data Designer (tests, export script)
    import data_designer.config as dd
except ImportError:  # pragma: no cover
    dd = None

# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #
def load_dotenv_nearby() -> Path | None:
    """Load the nearest ``.env`` (cwd upward, then this file upward) without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (base, *base.parents):
            cand = d / ".env"
            if cand.is_file():
                load_dotenv(cand, override=False)
                return cand
    return None


# --------------------------------------------------------------------------- #
# diversity axes
# --------------------------------------------------------------------------- #
DOMAIN_WEIGHTS = {
    "earth science": 0.22,
    "life science": 0.24,
    "physical science": 0.24,
    "space science": 0.10,
    "human body and health": 0.08,
    "environment and human impact": 0.08,
    "scientific method and measurement": 0.04,
}
DOMAIN_CODES = {
    "earth science": "ES", "life science": "LS", "physical science": "PS", "space science": "SP",
    "human body and health": "HB", "environment and human impact": "EN",
    "scientific method and measurement": "SM",
}
TOPICS = {
    "earth science": [
        "weather and climate", "the water cycle", "rocks and minerals", "weathering and erosion",
        "plate tectonics, earthquakes and volcanoes", "soil formation", "natural resources",
        "fossils and geologic time", "oceans and landforms",
    ],
    "life science": [
        "photosynthesis", "food chains and food webs", "animal adaptations and behavior",
        "plant structures and functions", "life cycles", "heredity and traits",
        "ecosystems and habitats", "cells and organisms", "classification of living things",
    ],
    "physical science": [
        "states of matter and phase changes", "energy transformations", "heat and temperature",
        "electricity and circuits", "magnetism", "forces and motion", "light and optics",
        "sound and waves", "physical vs chemical changes", "simple machines",
    ],
    "space science": [
        "the earth-sun-moon system", "seasons", "planets and the solar system",
        "stars and constellations", "day and night", "gravity and orbits",
    ],
    "human body and health": [
        "digestion and nutrition", "circulation and respiration", "the senses and the nervous system",
        "bones and muscles", "disease prevention",
    ],
    "environment and human impact": [
        "pollution", "conservation and recycling", "renewable and nonrenewable energy",
        "habitat change and extinction", "human effects on climate",
    ],
    "scientific method and measurement": [
        "units and measuring tools", "designing experiments and variables", "reading graphs and data",
        "classifying and observing",
    ],
}
QUESTION_STYLES = [
    "which of the following best describes ...",
    "what happens when ... (cause and effect)",
    "which is an example of ...",
    "why does ... / what causes ...",
    "which tool, material or object is best for a given task",
    "what is the purpose or function of ...",
    "predict the outcome of a described situation",
]

# Skewed toward 3+ steps (real data: 1 step 25%, 2 steps 25%, median 2). Chains past ~12 steps
# are rare in the real data and hard for an LLM to keep coherent, so the tail stops at 12.
STEP_WEIGHTS = {1: 0.02, 2: 0.10, 3: 0.22, 4: 0.18, 5: 0.14, 6: 0.10, 7: 0.08, 8: 0.06,
                9: 0.04, 10: 0.03, 11: 0.02, 12: 0.01}
FAN_IN = ((2, 0.90), (3, 0.09), (4, 0.01))        # real: 2 -> 90.7 %, 3 -> 8.1 %, 4+ -> 1.2 %
P_BRANCH_SMALL, P_BRANCH_LARGE = 0.50, 0.75       # branched share for 3-4 / 5+ steps (real: ~60 % of 3+)
P_REUSE_LEAF = 0.08                               # real: 8.2 % of proofs reuse a node


# --------------------------------------------------------------------------- #
# proof-graph helpers (pure)
# --------------------------------------------------------------------------- #
def _num(node: str) -> int:
    return int(re.sub(r"\D", "", node))


def sort_premises(premises: list[str]) -> list[str]:
    """Real convention (98 % of steps): derived ``intN`` first, then ``sentN``, each ascending."""
    ints = sorted((p for p in premises if p.startswith("int")), key=_num)
    sents = sorted((p for p in premises if p.startswith("sent")), key=_num)
    return ints + sents


def graph_stats(steps: list[dict]) -> dict:
    """Depth, shape and consistency facts for a step list (``conclusion``/``premises`` dicts)."""
    height: dict[str, int] = {}
    for s in steps:
        height[s["conclusion"]] = 1 + max((height.get(p, 0) for p in s["premises"]), default=0)
    n_int_prem = [sum(p.startswith("int") for p in s["premises"]) for s in steps]
    consumed = [p for s in steps for p in s["premises"]]
    n = len(steps)
    shape = "single-step" if n == 1 else ("branched" if any(k >= 2 for k in n_int_prem) else "linear")
    return {
        "n_steps": n,
        "depth": height.get(steps[-1]["conclusion"], 0) if steps else 0,
        "shape": shape,
        "n_leaves": len({p for p in consumed if p.startswith("sent")}),
        "fan_in": [len(s["premises"]) for s in steps],
        "join_steps": sum(k >= 2 for k in n_int_prem),
        "reused_nodes": sorted({p for p in consumed if consumed.count(p) > 1}, key=_num),
    }


def _weighted(rng: random.Random, pairs) -> int:
    r, acc = rng.random(), 0.0
    for value, p in pairs:
        acc += p
        if r < acc:
            return value
    return pairs[-1][0]


def build_skeleton(n_steps: int, rng: random.Random, *, p_reuse: float = P_REUSE_LEAF) -> dict:
    """Sample a proof graph with ``n_steps`` steps.

    Linear proofs are a spine (each step consumes the previous conclusion); branched proofs add
    ``1 + Binomial(n-3, .35)`` off-spine sub-derivations, guaranteeing at least one join. Steps
    are numbered in post-order (``int1``... in proof order, root = ``hypothesis`` last). Leaves
    get shuffled ``sentN`` ids, and with probability ``p_reuse`` one leaf feeds two steps.
    """
    n = max(1, int(n_steps))
    children: list[list[int]] = [[] for _ in range(n)]
    spine = n
    if n >= 2:
        branched = n >= 3 and rng.random() < (P_BRANCH_SMALL if n <= 4 else P_BRANCH_LARGE)
        extra = 1 + sum(rng.random() < 0.35 for _ in range(n - 3)) if branched else 0
        spine = n - extra
    root = spine - 1                                      # top of the spine = the hypothesis
    if n >= 2:
        for i in range(1, spine):
            children[i].append(i - 1)
        for v in range(spine, n):                         # off-spine sub-derivations
            def has_room(p):
                return len(children[p]) < 4
            if v == spine:                                 # first one: attach to a spine node that
                options = [p for p in range(1, spine) if has_room(p)]   # already has a child -> join
            elif rng.random() < 0.7:
                options = [p for p in range(spine) if has_room(p)]
            else:
                options = [p for p in range(v) if has_room(p)]
            children[rng.choice(options or [p for p in range(v) if has_room(p)] or [root])].append(v)

    # leaves per step
    leaf_slots: dict[int, int] = {}
    for v in range(n):
        k, c = _weighted(rng, FAN_IN), len(children[v])
        leaf_slots[v] = max(k, 2) if c == 0 else (max(k - 1, 1) if c == 1 else max(k - c, 0))

    order: list[int] = []                                  # post-order, children shuffled

    def visit(v: int) -> None:
        kids = list(children[v])
        rng.shuffle(kids)
        for c in kids:
            visit(c)
        order.append(v)

    visit(root)
    step_id = {v: (f"int{i + 1}" if v != root else "hypothesis") for i, v in enumerate(order)}

    leaf_idx: dict[int, list[int]] = {}
    nxt = 0
    for v in order:
        leaf_idx[v] = list(range(nxt, nxt + leaf_slots[v]))
        nxt += leaf_slots[v]

    if n >= 2 and rng.random() < p_reuse:                  # one leaf shared by two steps
        pairs = [(a, b) for a in order for b in order if a != b and leaf_idx[a] and leaf_idx[b]]
        rng.shuffle(pairs)
        for a, b in pairs:
            shareable = [x for x in leaf_idx[a] if x not in leaf_idx[b]]
            if shareable:
                leaf_idx[b][rng.randrange(len(leaf_idx[b]))] = rng.choice(shareable)
                break

    used = list(dict.fromkeys(x for v in order for x in leaf_idx[v]))
    perm = list(range(1, len(used) + 1))
    rng.shuffle(perm)
    leaf_id = {x: f"sent{perm[i]}" for i, x in enumerate(used)}

    steps = []
    for v in order:
        prem = [step_id[c] for c in children[v]] + [leaf_id[x] for x in leaf_idx[v]]
        steps.append({"conclusion": step_id[v], "premises": sort_premises(prem)})
    stats = graph_stats(steps)
    return {"steps": steps, "leaf_ids": [f"sent{i}" for i in range(1, len(used) + 1)],
            "int_ids": [f"int{i}" for i in range(1, n)], **stats}


def skeleton_text(skel: dict) -> str:
    """Human-readable skeleton for the generation prompt."""
    lines = []
    for i, s in enumerate(skel["steps"], 1):
        target = "HYPOTHESIS (the final conclusion)" if s["conclusion"] == "hypothesis" else s["conclusion"]
        lines.append(f"step {i}: {' & '.join(s['premises'])}  =>  {target}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# raw-record assembly (pure) -- reproduces the real proof encodings exactly
# --------------------------------------------------------------------------- #
def clean_sentence(text: str) -> str:
    """Normalise to the corpus convention: lowercase, single line, no ';' or ':', no final period."""
    t = re.sub(r"\s+", " ", str(text)).strip().lower()
    t = t.replace(";", ",").replace(":", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(". ").strip()


def _root_id(steps: list[dict]) -> str:
    return f"int{len(steps)}"


def _cid(step: dict, steps: list[dict]) -> str:
    return _root_id(steps) if step["conclusion"] == "hypothesis" else step["conclusion"]


def proof_string(steps: list[dict], int_text: dict[str, str]) -> str:
    chunks = []
    for s in steps:
        lhs = " & ".join(s["premises"])
        chunks.append(f"{lhs} -> hypothesis;" if s["conclusion"] == "hypothesis"
                      else f"{lhs} -> {s['conclusion']}: {int_text[s['conclusion']]};")
    return " ".join(chunks) + " "


def lisp_proof(steps: list[dict]) -> str:
    by_c = {s["conclusion"]: s for s in steps}

    def expr(s: dict) -> str:
        parts = [expr(by_c[p]) if p.startswith("int") else p for p in s["premises"]]
        return "((" + " ".join(parts) + ") -> " + _cid(s, steps) + ")"

    return expr(steps[-1])


def polish_proof(steps: list[dict]) -> str:
    """Prefix notation. Premises are space-joined, except that with 3+ premises all but the last two
    are separated by ' & ' (``# int1 & sent1 & sent3 sent5``). Matches 1,826 of the 1,840 real records;
    the rest are 1-premise steps and proofs that reuse an intermediate (not generated here)."""
    by_c = {s["conclusion"]: s for s in steps}

    def expr(s: dict) -> str:
        parts = [expr(by_c[p]) if p.startswith("int") else p for p in s["premises"]]
        body = " ".join(parts) if len(parts) <= 2 else " & ".join(parts[:-2]) + " & " + " ".join(parts[-2:])
        return "# " + _cid(s, steps) + " & " + body

    return expr(steps[-1])


def full_text_proof(steps: list[dict], leaf_text: dict[str, str], int_text: dict[str, str]) -> str:
    chunks = []
    for s in steps:
        prem = " [AND] ".join(leaf_text[p] if p.startswith("sent") else p for p in s["premises"])
        cid = _cid(s, steps)
        chunks.append(f"[BECAUSE] {prem} [INFER] {cid}: {int_text[cid]}")
    return " " + " ".join(chunks)


def proof_rendered(steps: list[dict], leaf_text: dict[str, str], int_text: dict[str, str],
                   hypothesis: str) -> str:
    """Readable proof for the LLM judge (ids shown next to their sentences)."""
    def label(p: str) -> str:
        return f"[{p}] {leaf_text[p]}" if p.startswith("sent") else f"[{p}] {int_text[p]}"

    out = []
    for i, s in enumerate(steps, 1):
        concl = f"[hypothesis] {hypothesis}" if s["conclusion"] == "hypothesis" else label(s["conclusion"])
        out.append(f"step {i}: " + "  AND  ".join(label(p) for p in s["premises"]) + f"\n   => {concl}")
    return "\n".join(out)


def assemble_task1(*, rid: str, question: str, answer: str, hypothesis: str, steps: list[dict],
                   leaf_text: dict[str, str], int_text: dict[str, str]) -> dict:
    """Build a raw task_1 record. ``leaf_text`` keys must be exactly ``sent1..sentK``; ``int_text``
    holds the non-root intermediates (the root's text is ``hypothesis``)."""
    stats = graph_stats(steps)
    root = _root_id(steps)
    ints = {**int_text, root: hypothesis}
    leaf_text = {f"sent{i}": leaf_text[f"sent{i}"] for i in range(1, len(leaf_text) + 1)}
    proof = proof_string(steps, int_text)
    return {
        "id": rid,
        "context": " ".join(f"{k}: {v}" for k, v in leaf_text.items()),
        "question": question,
        "answer": answer,
        "hypothesis": hypothesis,
        "proof": proof,
        "full_text_proof": full_text_proof(steps, leaf_text, ints),
        "depth_of_proof": stats["depth"],
        "length_of_proof": stats["n_steps"],
        "meta": {
            "question_text": question,
            "answer_text": answer,
            "hypothesis_id": root,
            "triples": dict(leaf_text),
            "distractors": [],
            "distractors_relevance": [],
            "intermediate_conclusions": dict(sorted(ints.items(), key=lambda kv: _num(kv[0]))),
            "core_concepts": [hypothesis],
            "step_proof": proof,
            "lisp_proof": lisp_proof(steps),
            "polish_proof": polish_proof(steps),
            # synthetic provenance: no WorldTree facts. Every leaf is "annotator-written", the same
            # convention the real data uses (uuid == "" <=> listed in add_list).
            "worldtree_provenance": {k: {"uuid": "", "original_text": v} for k, v in leaf_text.items()},
            "add_list": [{"sid": k, "fact": v} for k, v in leaf_text.items()],
            "delete_list": [],
        },
    }


_STEP_RE = re.compile(r"\s*([^;]*?)\s*->\s*(hypothesis|(int\d+):\s*([^;]*?))\s*;")


def parse_steps(proof: str) -> tuple[list[dict], dict[str, str]]:
    """Parse a ``proof`` string -> (steps, non-root intermediate texts)."""
    steps, ints = [], {}
    for m in _STEP_RE.finditer(proof):
        premises = [p.strip() for p in m.group(1).split("&") if p.strip()]
        if m.group(2) == "hypothesis":
            steps.append({"conclusion": "hypothesis", "premises": premises})
        else:
            steps.append({"conclusion": m.group(3), "premises": premises})
            ints[m.group(3)] = m.group(4)
    return steps, ints


def check_task1_record(rec: dict) -> list[str]:
    """Structural invariants of a task_1 record (the data-field and proof-logic checks from
    docs/entailmentbank_schema.md). Returns a list of problems; empty means consistent."""
    errs: list[str] = []
    try:
        meta = rec["meta"]
        triples = meta["triples"]
        steps, ints = parse_steps(rec["proof"])
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable record: {exc!r}"]
    if not steps:
        return ["empty proof"]
    if list(triples) != [f"sent{i}" for i in range(1, len(triples) + 1)]:
        errs.append("triples keys are not sent1..sentK")
    ctx = dict(re.findall(r"(sent\d+): (.*?)(?= sent\d+: |$)", rec["context"]))
    if ctx != triples:
        errs.append("context != meta.triples")
    if rec["proof"] != meta["step_proof"]:
        errs.append("proof != meta.step_proof")
    n = len(steps)
    if rec["length_of_proof"] != n:
        errs.append("length_of_proof != number of steps")
    if meta["hypothesis_id"] != f"int{n}":
        errs.append("hypothesis_id != int{n_steps}")
    ic = meta["intermediate_conclusions"]
    if ic.get(f"int{n}") != rec["hypothesis"]:
        errs.append("intermediate_conclusions[hypothesis_id] != hypothesis")
    if set(ic) != {f"int{i}" for i in range(1, n + 1)}:
        errs.append("intermediate_conclusions keys != int1..intN")
    if sum(s["conclusion"] == "hypothesis" for s in steps) != 1 or steps[-1]["conclusion"] != "hypothesis":
        errs.append("final step must be the only one concluding hypothesis")
    if [s["conclusion"] for s in steps[:-1]] != [f"int{i}" for i in range(1, n)]:
        errs.append("intermediates not numbered int1..int(n-1) in proof order")
    defined: set[str] = set()
    for s in steps:
        for p in s["premises"]:
            if p.startswith("int") and p not in defined:
                errs.append(f"{p} used before it is derived")
            elif not (p.startswith("int") or p in triples):
                errs.append(f"{p} is not a context sentence")
        if len(s["premises"]) < 2:
            errs.append("step with fewer than 2 premises")
        defined.add(s["conclusion"])
    used = {p for s in steps for p in s["premises"]}
    if {k for k in triples} - used:
        errs.append("unused leaf sentence")
    consumed_ints = {p for s in steps for p in s["premises"] if p.startswith("int")}
    if {c for c in defined if c.startswith("int")} - consumed_ints:
        errs.append("derived conclusion never used")
    for k, v in {**triples, **ic}.items():
        if not v or v != clean_sentence(v):
            errs.append(f"{k}: sentence not normalised")
    if len(set(triples.values())) < len(triples):
        errs.append("duplicate leaf sentences")
    if rec["hypothesis"] in triples.values():
        errs.append("a leaf equals the hypothesis")
    if not errs:
        g = graph_stats(steps)
        if rec["depth_of_proof"] != g["depth"]:
            errs.append("depth_of_proof != recomputed depth")
        ints_all = {k: v for k, v in ic.items() if k != f"int{n}"}
        regen = {
            "proof": proof_string(steps, ints_all),
            "lisp_proof": lisp_proof(steps),
            "polish_proof": polish_proof(steps),
            "full_text_proof": full_text_proof(steps, triples, ic),
        }
        got = {"proof": rec["proof"], "lisp_proof": meta["lisp_proof"], "polish_proof": meta["polish_proof"],
               "full_text_proof": rec["full_text_proof"]}
        errs += [f"{k} does not match the proof structure" for k in regen if regen[k] != got[k]]
    return errs


# --------------------------------------------------------------------------- #
# Data Designer pieces
# --------------------------------------------------------------------------- #
class Fact(BaseModel):
    id: str = Field(description="The id from the skeleton, exactly as written (e.g. sent3 or int2).")
    text: str = Field(description="One short declarative sentence: lowercase, no final period, "
                                  "no semicolons or colons, 4-16 words.")


class TreeContent(BaseModel):
    question: str = Field(description="A grade-school science question (capitalised, ends with '?' or a "
                                      "full stop), with no answer options.")
    answer: str = Field(description="The short correct answer (1-8 words), capitalised like a sentence.")
    hypothesis: str = Field(description="The question and answer restated as ONE declarative sentence "
                                        "(lowercase, no final period). It is the final conclusion.")
    leaves: list[Fact] = Field(description="One fact per leaf id in the skeleton.")
    intermediates: list[Fact] = Field(description="One conclusion per intermediate id in the skeleton "
                                                  "(NOT the hypothesis).")


def _rng_for(record_uuid: str) -> random.Random:
    return random.Random(int(hashlib.sha256(str(record_uuid).encode()).hexdigest(), 16))


def _identity(fn):
    return fn


_custom = dd.custom_column_generator if dd else (lambda **_: _identity)


@_custom(required_columns=["record_uuid", "n_steps"],
         side_effect_columns=["skeleton_text", "leaf_ids_text", "int_ids_text", "reused_text"])
def make_skeleton(row: dict) -> dict:
    # Data Designer's Jinja sandbox has no `join` filter, so the prompt gets ready-made strings.
    skel = build_skeleton(int(row["n_steps"]), _rng_for(row["record_uuid"]))
    row["skeleton"] = skel
    row["skeleton_text"] = skeleton_text(skel)
    row["leaf_ids_text"] = ", ".join(skel["leaf_ids"])
    row["int_ids_text"] = ", ".join(skel["int_ids"]) or "(none: a single step)"
    row["reused_text"] = (
        f"These ids feed more than one step, so their fact must be useful in each: {', '.join(skel['reused_nodes'])}.\n"
        if skel["reused_nodes"] else "")
    return row


def _as_obj(x):
    """Parquet/JSON round-trips can hand back JSON strings, numpy arrays or None."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return {}
    return x if x is not None else {}


def build_record(record_uuid, domain: str, skel, tree) -> tuple[dict, str, list[str]]:
    """LLM sentences + sampled skeleton -> (raw task_1 record, readable proof, problems).

    Pure and deterministic, so the export script can rebuild exactly the same record from the
    saved ``skeleton`` and ``tree`` columns."""
    skel, tree = _as_obj(skel), _as_obj(tree)
    errors: list[str] = []
    leaves = {str(f["id"]).strip(): clean_sentence(f["text"]) for f in (tree.get("leaves") or [])}
    inter = {str(f["id"]).strip(): clean_sentence(f["text"]) for f in (tree.get("intermediates") or [])}
    steps = [{"conclusion": s["conclusion"], "premises": list(s["premises"])} for s in skel["steps"]]
    if set(leaves) != set(skel["leaf_ids"]):
        errors.append(f"leaf ids {sorted(leaves)} != skeleton {list(skel['leaf_ids'])}")
    if set(inter) != set(skel["int_ids"]):
        errors.append(f"intermediate ids {sorted(inter)} != skeleton {list(skel['int_ids'])}")
    question = re.sub(r"\s+", " ", str(tree.get("question", ""))).strip()
    answer = re.sub(r"\s+", " ", str(tree.get("answer", ""))).strip()
    hypothesis = clean_sentence(tree.get("hypothesis", ""))
    if not (question and answer and hypothesis):
        errors.append("missing question/answer/hypothesis")
    if errors:
        return {}, "", errors
    rec = assemble_task1(rid=f"SYN_{DOMAIN_CODES.get(domain, 'XX')}_{str(record_uuid)[:8]}", question=question,
                         answer=answer, hypothesis=hypothesis, steps=steps, leaf_text=leaves, int_text=inter)
    return rec, proof_rendered(steps, leaves, inter, hypothesis), check_task1_record(rec)


@_custom(required_columns=["record_uuid", "domain", "skeleton", "tree"],
         side_effect_columns=["proof_text", "structure_valid", "structure_errors"])
def assemble_record(row: dict) -> dict:
    rec, proof_txt, errors = build_record(row["record_uuid"], row["domain"], row["skeleton"], row.get("tree"))
    row["record"] = rec
    row["proof_text"] = proof_txt
    row["structure_valid"] = not errors
    row["structure_errors"] = "; ".join(errors)
    return row


def _model_name(var: str, fallback: str | None = None) -> str:
    name = (os.environ.get(var) or (os.environ.get(fallback) if fallback else "") or "").strip()
    if not name:
        raise RuntimeError(f"{var} is not set (put it in .env or the environment)")
    return name.removeprefix("openai/")


GEN_ALIAS, JUDGE_ALIAS = "eb-synth-gen", "eb-synth-judge"

GEN_SYSTEM = """You write EntailmentBank-style entailment trees for grade-school science questions.
An entailment tree derives a hypothesis (the answer to a question, as a statement) from simple
facts, one small logical step at a time. Rules:
- Leaf facts are short, general, textbook-style statements ("an animal is a kind of organism",
  "friction causes a moving object to slow down", "a plant needs sunlight to make food").
  They must be true, and must NOT mention the specific question scenario unless a fact is a
  scenario-specific observation the question itself provides.
- Each step's conclusion must follow from its premises TAKEN TOGETHER, using only what they say plus
  common sense. Every premise must be needed; no step may just restate a premise.
- All sentences: lowercase, no final period, no semicolons or colons.
- The final conclusion (the hypothesis) restates the question's answer as one declarative sentence.
- Never reuse the same sentence for two different ids."""

GEN_PROMPT = """Write one entailment tree.

domain: {{ domain }}
topic: {{ topic }}
question style: {{ question_style }}

Proof skeleton (the shape is fixed; you supply the sentences). Premises are joined with '&'.
{{ skeleton_text }}

Leaf ids (write one fact for each): {{ leaf_ids_text }}
Intermediate ids (write one conclusion for each, in order): {{ int_ids_text }}
{{ reused_text }}
Write a question about the topic whose answer is the hypothesis, then the facts and conclusions so
that every step is a valid entailment. Return every id exactly as written above.

Example of the style (skeleton: sent2 & sent5 => int1; sent1 & sent3 => int2; int1 & int2 => int3; int3 & sent4 => hypothesis):
question: An earthquake can change the surface of the Earth. How does an earthquake change it?
answer: Quickly
sent1: shaking the ground of the earth can change earth's surface
sent2: 10 to 30 seconds is a short amount of time
sent3: an earthquake causes the ground to shake
sent4: in a short amount of time is similar to quickly
sent5: an earthquake usually occurs over a period of 10 to 30 seconds
int1: an earthquake usually occurs in a short amount of time
int2: an earthquake can change earth's surface by shaking the ground
int3: an earthquake can change earth's surface in a short amount of time
hypothesis: an earthquake changes earth's surface quickly"""

JUDGE_PROMPT = """You are checking an entailment tree for grade-school science.

domain: {{ domain }}
question: {{ record.question }}
answer: {{ record.answer }}
hypothesis: {{ record.hypothesis }}

Proof (each step says its premises, taken together, entail the conclusion):
{{ proof_text }}"""


def _scores():
    five = {5: "excellent", 4: "good", 3: "borderline", 2: "poor", 1: "wrong"}
    return [
        dd.Score(name="step_validity", description=(
            "For EVERY step, do the premises taken together logically entail the conclusion, using only what "
            "they state plus common sense? Penalise leaps, missing premises and conclusions that need an "
            "unstated fact."),
            options={5: "every step follows strictly", 4: "every step follows; one is slightly loose",
                     3: "one step needs a clearly unstated assumption", 2: "several steps do not follow",
                     1: "the chain is mostly non-sequiturs"}),
        dd.Score(name="factual_correctness", description=(
            "Are all facts and conclusions true as grade-school science?"), options=five),
        dd.Score(name="answer_consistency", description=(
            "Does the hypothesis faithfully state the question's answer as a declarative sentence, and is "
            "that answer actually correct for the question?"), options=five),
        dd.Score(name="non_triviality", description=(
            "Is every premise needed, and does each conclusion say something new rather than restating a "
            "premise or the hypothesis?"),
            options={5: "every premise needed, every step adds content", 4: "mostly; one mild redundancy",
                     3: "some premises or steps are redundant", 2: "many are redundant", 1: "mostly restatement"}),
    ]


def load_config_builder() -> "dd.DataDesignerConfigBuilder":
    load_dotenv_nearby()
    gen_model = _model_name("OPENAI_MODEL")
    judge_model = _model_name("OPENAI_JUDGE_MODEL", fallback="OPENAI_MODEL")

    model_configs = [
        dd.ModelConfig(alias=GEN_ALIAS, model=gen_model, provider="openai",
                       inference_parameters=dd.ChatCompletionInferenceParams(max_parallel_requests=8)),
        dd.ModelConfig(alias=JUDGE_ALIAS, model=judge_model, provider="openai",
                       inference_parameters=dd.ChatCompletionInferenceParams(max_parallel_requests=8)),
    ]
    b = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    # ---- axes of diversity ------------------------------------------------- #
    b.add_column(dd.SamplerColumnConfig(name="record_uuid", sampler_type="uuid",
                                        params=dd.UUIDSamplerParams(short_form=True)))
    b.add_column(dd.SamplerColumnConfig(
        name="domain", sampler_type="category",
        params=dd.CategorySamplerParams(values=list(DOMAIN_WEIGHTS), weights=list(DOMAIN_WEIGHTS.values()))))
    b.add_column(dd.SamplerColumnConfig(
        name="topic", sampler_type="subcategory",
        params=dd.SubcategorySamplerParams(category="domain", values=TOPICS)))
    b.add_column(dd.SamplerColumnConfig(
        name="question_style", sampler_type="category", params=dd.CategorySamplerParams(values=QUESTION_STYLES)))
    b.add_column(dd.SamplerColumnConfig(
        name="n_steps", sampler_type="category",
        params=dd.CategorySamplerParams(values=list(STEP_WEIGHTS), weights=list(STEP_WEIGHTS.values()))))

    # ---- structure first, then text ---------------------------------------- #
    b.add_column(dd.CustomColumnConfig(name="skeleton", generator_function=make_skeleton))
    b.add_column(dd.LLMStructuredColumnConfig(
        name="tree", model_alias=GEN_ALIAS, system_prompt=GEN_SYSTEM, prompt=GEN_PROMPT,
        output_format=TreeContent))
    b.add_column(dd.CustomColumnConfig(name="record", generator_function=assemble_record))

    # ---- quality scores (kept as columns; the export script filters) ------- #
    b.add_column(dd.LLMJudgeColumnConfig(
        name="judge", model_alias=JUDGE_ALIAS, prompt=JUDGE_PROMPT, scores=_scores(),
        skip=dd.SkipConfig(when="{{ not structure_valid }}")))
    return b


if __name__ == "__main__":  # quick look at the sampled skeleton distribution, no API calls
    rng = random.Random(0)
    import collections
    shapes, steps = collections.Counter(), collections.Counter()
    for i in range(5000):
        n = _weighted(rng, list(STEP_WEIGHTS.items()))
        sk = build_skeleton(n, random.Random(i))
        shapes[sk["shape"]] += 1
        steps[sk["n_steps"]] += 1
    print("shapes", dict(shapes), "\nsteps", dict(sorted(steps.items())), file=sys.stdout)
