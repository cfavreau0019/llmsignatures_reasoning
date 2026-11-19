"""Turn a Data Designer run into EntailmentBank-format files (task_1, task_2, task_3).

    .venv/Scripts/python scripts/export_synthetic_entailmentbank.py \
        --input <path to the dataset dir / *.parquet from `data-designer create` or `preview`> \
        --out   data/synthetic/entailmentbank_synth_v1

What it does
------------
1. Reads the Data Designer rows and **rebuilds each task_1 record** from the saved ``skeleton`` and
   ``tree`` columns with the same pure code the generator used (no reliance on nested-parquet
   round-trips), then re-checks every structural invariant.
2. Filters: structurally valid, every judge score >= ``--min-judge-score`` (default 4),
   no duplicate question/hypothesis. Nothing is thrown away silently -- reasons are counted in
   ``manifest.json``. Use ``--keep-all`` to skip the judge filter, and ``--rejects [PATH]`` to also
   write every dropped row (reason, judge scores and rationales, the rebuilt record, the raw LLM
   tree) to a JSONL file for failure analysis.
3. Builds the other tasks, which need cross-record context:
   * task_2: gold leaves + distractors drawn from *other* synthetic records (60 % same domain),
     always 25 sentences, leaf ids renumbered, proof re-encoded.
   * task_3: 25 sentences that contain only ~77 % of the gold leaves on average (the real coverage),
     ``proof == ""``, ``" / "`` written as ``"; "`` like the real task_3 text.
4. Validates every output record against ``docs/schemas/*.schema.json`` and writes
   ``<out>/dataset/task_N/{train,dev,test}.jsonl`` (+ ``task_3/all.jsonl``), so
   ``EntailmentBank(local_dir=<out>)`` loads it. ``manifest.json`` holds counts, QC stats and
   per-record judge scores.

Synthetic records use ids ``SYN_<domain>_<hex>`` and live in their own directory: they are never
mixed into the real train/dev/test splits.

Synthetic stand-ins (documented in manifest.json): ``worldtree_provenance.uuid`` is ``""`` and
``add_list`` lists every leaf (the real convention for annotator-written facts); ``delete_list`` is
empty; ``distractors_relevance`` / ``relevance_score`` are a lexical-overlap score in [0, 1], NOT the
retrieval scores of the real release and not comparable to them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_entailmentbank as G  # noqa: E402  (pure helpers; no Data Designer needed)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "docs" / "schemas"
SPLITS = ("train", "dev", "test")
CONTEXT_SIZE = 25
JUDGE_SCORES = ("step_validity", "factual_correctness", "answer_consistency", "non_triviality")


# --------------------------------------------------------------------------- #
# reading Data Designer output
# --------------------------------------------------------------------------- #
def to_py(x):
    """numpy arrays / scalars / NaN -> plain python."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def load_rows(path: Path) -> list[dict]:
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.parquet"))
        files = [f for f in files if "processor" not in f.as_posix()] or files
        jsonl = sorted(path.rglob("*.jsonl"))
        if not files and jsonl:
            return [json.loads(l) for f in jsonl for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        files = [path]
    if not files:
        sys.exit(f"no .parquet / .jsonl found under {path}")
    rows: list[dict] = []
    for f in files:
        rows += [to_py(r) for r in pd.read_parquet(f).to_dict("records")]
    return rows


def judge_scores(row: dict) -> dict[str, int | None]:
    j = G._as_obj(row.get("judge"))
    out = {}
    for name in JUDGE_SCORES:
        v = j.get(name) if isinstance(j, dict) else None
        v = v.get("score") if isinstance(v, dict) else v
        try:
            out[name] = int(v)
        except (TypeError, ValueError):
            out[name] = None
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rng_for(seed: int, key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16))


def split_of(rid: str, fractions: tuple[float, float, float]) -> str:
    u = int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "train" if u < fractions[0] else ("dev" if u < fractions[0] + fractions[1] else "test")


_TOK = re.compile(r"[a-z0-9']+")


def lexical_score(text: str, reference: str) -> float:
    a, b = set(_TOK.findall(text.lower())), set(_TOK.findall(reference.lower()))
    return round(len(a & b) / len(a | b), 4) if a | b else 0.0


class DistractorPool:
    """Leaf sentences of the kept records, indexed by domain, to serve as distractors."""

    def __init__(self, records: list[dict], domains: dict[str, str]):
        self.items: list[tuple[str, str, str]] = []          # (text, domain, record id)
        for r in records:
            for t in r["meta"]["triples"].values():
                self.items.append((t, domains.get(r["id"], ""), r["id"]))
        self.by_domain: dict[str, list[tuple[str, str]]] = {}
        for t, d, rid in self.items:
            self.by_domain.setdefault(d, []).append((t, rid))

    def draw(self, rng: random.Random, n: int, *, domain: str, exclude_rid: str, exclude_texts: set[str],
             same_domain_frac: float = 0.6) -> list[str]:
        out: list[str] = []
        seen = set(exclude_texts)

        def take(cands: list[tuple[str, str]], k: int) -> None:
            pool = [t for t, rid in cands if rid != exclude_rid and t not in seen]
            rng.shuffle(pool)
            for t in pool:
                if k <= 0:
                    break
                if t not in seen:
                    seen.add(t); out.append(t); k -= 1

        take(self.by_domain.get(domain, []), round(n * same_domain_frac))
        take([(t, rid) for t, _, rid in self.items], n - len(out))
        return out if len(out) == n else []


def renumber(steps: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [{"conclusion": s["conclusion"],
             "premises": G.sort_premises([mapping.get(p, p) for p in s["premises"]])} for s in steps]


# --------------------------------------------------------------------------- #
# task builders
# --------------------------------------------------------------------------- #
def to_task2(rec: dict, pool: DistractorPool, domain: str, rng: random.Random) -> dict | None:
    steps, ints = G.parse_steps(rec["proof"])
    gold = rec["meta"]["triples"]
    need = CONTEXT_SIZE - len(gold)
    if need < 8:
        return None
    distract = pool.draw(rng, need, domain=domain, exclude_rid=rec["id"], exclude_texts=set(gold.values()))
    if not distract:
        return None
    texts = list(gold.values()) + distract
    order = list(range(len(texts)))
    rng.shuffle(order)                                          # position -> original index
    new_id = {oi: f"sent{pos + 1}" for pos, oi in enumerate(order)}
    mapping = {old: new_id[i] for i, old in enumerate(gold)}
    new_steps = renumber(steps, mapping)
    triples = {f"sent{pos + 1}": texts[oi] for pos, oi in enumerate(order)}
    gold_ids = set(mapping.values())
    leaf_text = {nid: triples[nid] for nid in gold_ids}
    ic = rec["meta"]["intermediate_conclusions"]
    ints_nr = {k: v for k, v in ic.items() if k != rec["meta"]["hypothesis_id"]}
    proof = G.proof_string(new_steps, ints_nr)
    ref = rec["hypothesis"] + " " + rec["question"]
    dist_ids = [k for k in triples if k not in gold_ids]
    sort_key = lambda k: int(k[4:])  # noqa: E731
    return {
        "id": rec["id"],
        "context": " ".join(f"{k}: {v}" for k, v in triples.items()),
        "question": rec["question"], "answer": rec["answer"], "hypothesis": rec["hypothesis"],
        "proof": proof,
        "full_text_proof": G.full_text_proof(new_steps, leaf_text, ic),
        "depth_of_proof": rec["depth_of_proof"], "length_of_proof": rec["length_of_proof"],
        "meta": {
            "question_text": rec["question"], "answer_text": rec["answer"],
            "hypothesis_id": rec["meta"]["hypothesis_id"],
            "triples": triples,
            "distractors": sorted(dist_ids, key=sort_key),
            "distractors_relevance": [lexical_score(triples[k], ref) for k in sorted(dist_ids, key=sort_key)],
            "intermediate_conclusions": dict(ic),
            "core_concepts": list(rec["meta"]["core_concepts"]),
            "step_proof": proof,
            "lisp_proof": G.lisp_proof(new_steps), "polish_proof": G.polish_proof(new_steps),
            "worldtree_provenance": {k: {"uuid": "", "original_text": leaf_text[k]}
                                     for k in sorted(gold_ids, key=sort_key)},
            "add_list": [{"sid": k, "fact": leaf_text[k]} for k in sorted(gold_ids, key=sort_key)],
            "delete_list": [],
        },
    }


def to_task3(rec: dict, pool: DistractorPool, domain: str, rng: random.Random, dup_prob: float) -> tuple[dict, float] | None:
    gold = list(rec["meta"]["triples"].values())
    cov = 1.0 if rng.random() < 0.30 else rng.uniform(0.35, 0.98)    # real: mean 0.77, 30 % full
    n_present = max(0, min(len(gold), round(cov * len(gold))))
    present = rng.sample(gold, n_present)
    distract = pool.draw(rng, CONTEXT_SIZE - n_present, domain=domain, exclude_rid=rec["id"],
                         exclude_texts=set(gold))
    if not distract:
        return None
    texts = present + distract
    if dup_prob and rng.random() < dup_prob and len(distract) >= 2:       # retrieval often returns repeats
        texts[len(present) + rng.randrange(len(distract))] = rng.choice(texts[: len(texts) - 1] or texts)
    rng.shuffle(texts)
    texts = [t.replace(" / ", "; ") for t in texts]                       # real task_3 text convention
    triples = {f"sent{i + 1}": t for i, t in enumerate(texts)}
    ref = rec["hypothesis"] + " " + rec["question"]
    return {
        "id": rec["id"],
        "context": " ".join(f"{k}: {v}" for k, v in triples.items()),
        "question": rec["question"], "answer": rec["answer"], "hypothesis": rec["hypothesis"],
        "proof": "",
        "meta": {
            "question_text": rec["question"], "answer_text": rec["answer"],
            "triples": triples,
            "core_concepts": list(rec["meta"]["core_concepts"]),
            "worldtree_provenance": [{"sid": k, "uuid": "", "relevance_score": lexical_score(v, ref)}
                                     for k, v in triples.items()],
        },
    }, (n_present / len(gold) if gold else 0.0)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def make_validators():
    from jsonschema import Draft202012Validator
    return {t: Draft202012Validator(json.loads((SCHEMAS / f"entailmentbank_task{t}_record.schema.json").read_text(encoding="utf-8")))
            for t in (1, 2, 3)}


def task2_problems(t2: dict, t1: dict) -> list[str]:
    errs = []
    tri, dist = t2["meta"]["triples"], t2["meta"]["distractors"]
    if len(tri) != CONTEXT_SIZE:
        errs.append("task2 context is not 25 sentences")
    steps, _ = G.parse_steps(t2["proof"])
    gold = {p for s in steps for p in s["premises"] if p.startswith("sent")}
    if not gold <= set(tri) or gold & set(dist):
        errs.append("task2 gold ids not in context / overlap distractors")
    if {tri[g] for g in gold} != set(t1["meta"]["triples"].values()):
        errs.append("task2 gold text set != task1 leaf set")
    if len(set(tri.values())) != len(tri):
        errs.append("task2 context has duplicate sentences")
    if t2["proof"] != t2["meta"]["step_proof"]:
        errs.append("task2 proof != step_proof")
    return errs


def real_sets(real_dir: Path | None) -> tuple[set[str], set[str], set[str]]:
    if real_dir is None:
        return set(), set(), set()
    leaves: set[str] = set(); hyps: set[str] = set(); qs: set[str] = set()
    for sp in SPLITS:
        f = real_dir / "dataset" / "task_1" / f"{sp}.jsonl"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    leaves |= set(r["meta"]["triples"].values()); hyps.add(r["hypothesis"]); qs.add(r["question"])
    return leaves, hyps, qs


def find_real_dir(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    rel = Path("data/raw/entailment_trees_emnlp2021_data_v3")
    for base in (Path.cwd(), *list(Path.cwd().parents)[:6]):
        if (base / rel / "dataset").is_dir():
            return base / rel
    return None


def reject_row(row: dict, reason: str, rec: dict | None, errs: list | None) -> dict:
    """One line of the rejects file: why the row was dropped plus everything needed to inspect it."""
    return {
        "reason": reason,
        "record_uuid": row.get("record_uuid"),
        "id": rec["id"] if rec else None,
        "domain": row.get("domain"), "topic": row.get("topic"),
        "question_style": row.get("question_style"), "n_steps": row.get("n_steps"),
        "judge_scores": judge_scores(row),
        "judge_raw": G._as_obj(row.get("judge")),             # includes the judge's reasoning text
        "structure_errors": errs if errs is not None else to_py(row.get("structure_errors")),
        "record": rec,                                         # rebuilt task_1 record (None if it could not be built)
        "tree": G._as_obj(row.get("tree")),                    # the LLM's raw sentences
        "skeleton_text": row.get("skeleton_text"),
    }


def summarize(xs) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": xs[0], "median": q(.5), "mean": round(st.mean(xs), 3), "p95": q(.95), "max": xs[-1]}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Data Designer output dir or .parquet/.jsonl file")
    ap.add_argument("--out", default=str(ROOT / "data/synthetic/entailmentbank_synth_v1"))
    ap.add_argument("--min-judge-score", type=int, default=4, help="every judge score must be >= this (1-5)")
    ap.add_argument("--keep-all", action="store_true", help="ignore the judge filter (structure checks still apply)")
    ap.add_argument("--split-fractions", default="0.8,0.1,0.1", help="train,dev,test")
    ap.add_argument("--task3-dup-prob", type=float, default=0.0,
                    help="chance a task_3 context repeats one sentence (real: 87%% of contexts do)")
    ap.add_argument("--real-data", default=None, help="real EntailmentBank dir, for overlap QC (auto-detected)")
    ap.add_argument("--drop-real-overlap", action="store_true",
                    help="drop records whose question or hypothesis text exists in the real data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rejects", nargs="?", const="", default=None, metavar="PATH",
                    help="write every dropped row (reason, judge scores + rationales, rebuilt record, raw tree) "
                         "to this JSONL file; with no PATH uses <out>/rejects.jsonl")
    ap.add_argument("--no-schema", action="store_true", help="skip JSON Schema validation")
    a = ap.parse_args()
    fr = tuple(float(x) for x in a.split_fractions.split(","))
    assert len(fr) == 3 and abs(sum(fr) - 1) < 1e-6, "--split-fractions must be three numbers summing to 1"

    rows = load_rows(Path(a.input))
    if a.limit:
        rows = rows[: a.limit]
    drops: Counter = Counter()
    kept: list[dict] = []
    meta: dict[str, dict] = {}
    seen_q: set[str] = set(); seen_h: set[str] = set(); seen_id: set[str] = set()
    real_leaves, real_hyps, real_qs = real_sets(find_real_dir(a.real_data))

    rejects: list[dict] = []
    rows_by_id: dict[str, dict] = {}                           # kept record id -> its Data Designer row

    def reject(row: dict, reason: str, rec: dict | None = None, errs: list | None = None) -> None:
        drops[reason] += 1
        rejects.append(reject_row(row, reason, rec, errs))

    for row in rows:
        rec, _, errs = G.build_record(row.get("record_uuid"), row.get("domain", ""), row.get("skeleton"), row.get("tree"))
        if errs:
            reject(row, "structure/consistency", None, errs)
            continue
        js = judge_scores(row)
        if not a.keep_all:
            if any(v is None for v in js.values()):
                reject(row, "no judge scores", rec)
                continue
            if min(js.values()) < a.min_judge_score:
                reject(row, f"judge < {a.min_judge_score}", rec)
                continue
        if rec["id"] in seen_id or rec["question"] in seen_q or rec["hypothesis"] in seen_h:
            reject(row, "duplicate id/question/hypothesis", rec)
            continue
        if a.drop_real_overlap and (rec["question"] in real_qs or rec["hypothesis"] in real_hyps):
            reject(row, "question/hypothesis exists in real data", rec)
            continue
        seen_id.add(rec["id"]); seen_q.add(rec["question"]); seen_h.add(rec["hypothesis"])
        leaves = list(rec["meta"]["triples"].values())
        meta[rec["id"]] = {
            "domain": row.get("domain"), "topic": row.get("topic"), "question_style": row.get("question_style"),
            "judge": js,
            "real_leaf_overlap": round(sum(t in real_leaves for t in leaves) / len(leaves), 3) if real_leaves else None,
        }
        kept.append(rec)
        rows_by_id[rec["id"]] = row

    print(f"read {len(rows)} rows -> kept {len(kept)}; dropped {dict(drops)}")
    if not kept:
        sys.exit("nothing to export")

    domains = {rid: m["domain"] or "" for rid, m in meta.items()}
    pool = DistractorPool(kept, domains)
    out = Path(a.out)
    files: dict[tuple[int, str], list[dict]] = {}
    t2_skipped = t3_skipped = 0
    coverage: list[float] = []
    validators = None if a.no_schema else make_validators()
    problems: list[str] = []

    for rec in kept:
        rid = rec["id"]
        rng = rng_for(a.seed, rid)
        t2 = to_task2(rec, pool, domains[rid], rng)
        t3 = to_task3(rec, pool, domains[rid], rng, a.task3_dup_prob)
        if t2 is None or t3 is None:                          # pool too small for 25 distinct sentences
            t2_skipped += t2 is None; t3_skipped += t3 is None
            reject(rows_by_id.get(rid, {"record_uuid": rid}), "pool too small for 25-sentence context", rec)
            continue
        t3_rec, cov = t3
        coverage.append(cov)
        problems += [f"{rid}: {e}" for e in task2_problems(t2, rec)]
        if validators:
            for t, r in ((1, rec), (2, t2), (3, t3_rec)):
                for e in list(validators[t].iter_errors(r))[:1]:
                    problems.append(f"{rid} task{t}: {e.message[:160]} @ {list(e.absolute_path)}")
        sp = split_of(rid, fr)
        meta[rid]["split"] = sp
        for t, r in ((1, rec), (2, t2), (3, t3_rec)):
            files.setdefault((t, sp), []).append(r)

    if problems:
        print(f"{len(problems)} problems, first 10:")
        for p in problems[:10]:
            print("  ", p)
        sys.exit("export aborted: fix the generator or filter these records")

    for (t, sp), recs in files.items():
        d = out / "dataset" / f"task_{t}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    for sp in SPLITS:                                          # loader needs every split file
        for t in (1, 2, 3):
            f = out / "dataset" / f"task_{t}" / f"{sp}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch(exist_ok=True)
    (out / "dataset" / "task_3" / "all.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for sp in SPLITS for r in files.get((3, sp), [])),
        encoding="utf-8")

    rejects_path = None
    if a.rejects is not None:
        rejects_path = Path(a.rejects) if a.rejects else out / "rejects.jsonl"
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        rejects_path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rejects),
                                encoding="utf-8")
        print(f"wrote {len(rejects)} rejected rows -> {rejects_path}")

    final = [r for sp in SPLITS for r in files.get((1, sp), [])]
    stats = [G.graph_stats(G.parse_steps(r["proof"])[0]) for r in final]
    manifest = {
        "generator": "scripts/synthetic_entailmentbank.py (NVIDIA Data Designer)",
        "args": vars(a),
        "rows_read": len(rows), "kept_after_filters": len(kept), "exported": len(final),
        "dropped": dict(drops), "rejects_file": str(rejects_path) if rejects_path else None,
        "task2_skipped_small_pool": t2_skipped, "task3_skipped_small_pool": t3_skipped,
        "split_counts": {sp: len(files.get((1, sp), [])) for sp in SPLITS},
        "proof_graph": {
            "n_steps": dict(sorted(Counter(s["n_steps"] for s in stats).items())),
            "depth": dict(sorted(Counter(s["depth"] for s in stats).items())),
            "shape": dict(Counter(s["shape"] for s in stats)),
            "n_leaves": summarize([s["n_leaves"] for s in stats]),
            "reused_node_proofs": sum(bool(s["reused_nodes"]) for s in stats),
            "usable_for_corruption_pairs(>=3 steps)": sum(s["n_steps"] >= 3 for s in stats),
        },
        "judge_mean": {k: round(st.mean(m["judge"][k] for m in meta.values() if m["judge"][k] is not None), 3)
                       for k in JUDGE_SCORES if any(m["judge"][k] is not None for m in meta.values())},
        "domain_counts": dict(Counter(m["domain"] for m in meta.values())),
        "task3_gold_coverage": summarize([round(c, 3) for c in coverage]),
        "task2_distractors_per_record": summarize([len(r["meta"]["distractors"]) for sp in SPLITS for r in files.get((2, sp), [])]),
        "real_data_overlap": {
            "records_with_any_real_leaf": sum(1 for m in meta.values() if m["real_leaf_overlap"]),
            "mean_leaf_overlap": round(st.mean(m["real_leaf_overlap"] for m in meta.values()
                                               if m["real_leaf_overlap"] is not None), 4) if real_leaves else None,
            "question_in_real": sum(r["question"] in real_qs for r in final) if real_qs else None,
            "hypothesis_in_real": sum(r["hypothesis"] in real_hyps for r in final) if real_hyps else None,
        },
        "synthetic_stand_ins": {
            "worldtree_provenance.uuid": "\"\" for every leaf; add_list lists every leaf (real convention for written facts)",
            "delete_list": "[]",
            "core_concepts": "[hypothesis]",
            "distractors_relevance / relevance_score": "lexical Jaccard overlap in [0,1] with hypothesis+question; "
                                                       "NOT comparable to the real retrieval scores",
            "task3 text": "' / ' written as '; ' like the real task_3; duplicate sentences only if --task3-dup-prob > 0",
        },
        "records": {rid: m for rid, m in meta.items() if "split" in m},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(final)} records x 3 tasks to {out}  (splits {manifest['split_counts']})")


if __name__ == "__main__":
    main()
