"""Profile the EntailmentBank v3 release and emit machine-readable statistics.

Produces ``docs/schemas/entailmentbank_stats.json`` -- the numbers quoted in
``docs/entailmentbank_schema.md``. Re-run after changing the loader or the data:

    python scripts/profile_entailmentbank.py --data data/raw/entailment_trees_emnlp2021_data_v3

Sections
--------
* ``fields``    -- every JSON path per task: type(s), presence, null/empty rate, example.
* ``graph``     -- proof-graph structure (depth, steps, fan-in, branching, ordering).
* ``integrity`` -- cross-field and cross-task consistency checks.
* ``pipeline``  -- what the repo's ``EntailmentBank`` loader actually produces.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_trajectories.datasets.entailmentbank import (  # noqa: E402
    EntailmentBank, build_id2text, parse_context, parse_proof_steps,
)

TASKS = (1, 2, 3)
SPLITS = ("train", "dev", "test")
VARIABLE_KEY_PARENTS = ("triples", "intermediate_conclusions", "worldtree_provenance")


# --------------------------------------------------------------------------- #
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(xs) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
    return {"n": n, "min": xs[0], "p25": q(.25), "median": q(.5), "mean": round(st.mean(xs), 3),
            "p75": q(.75), "p95": q(.95), "max": xs[-1]}


def hist(xs) -> dict:
    return {str(k): v for k, v in sorted(collections.Counter(xs).items())}


def jtype(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def walk(v, path=""):
    yield path, v
    if isinstance(v, dict):
        for k, x in v.items():
            seg = "*" if path.split(".")[-1] in VARIABLE_KEY_PARENTS else k
            yield from walk(x, f"{path}.{seg}" if path else seg)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x, f"{path}[]")


def infer_fields(rows: list[dict]) -> dict:
    types = collections.defaultdict(collections.Counter)
    present = collections.Counter()
    empty = collections.Counter()
    example: dict[str, object] = {}
    for r in rows:
        seen = set()
        for path, v in walk(r):
            if not path:
                continue
            types[path][jtype(v)] += 1
            seen.add(path)
            if v in ("", [], {}):
                empty[path] += 1
            example.setdefault(path, v if not isinstance(v, (dict, list)) or len(json.dumps(v)) < 120 else None)
        present.update(seen)
    out = {}
    for p in sorted(types):
        out[p] = {"types": dict(types[p]), "records_with_path": present[p], "empty_values": empty[p],
                  "example": example[p]}
    return out


# --------------------------------------------------------------------------- #
# proof-graph analysis
# --------------------------------------------------------------------------- #
def analyze_tree(rec: dict) -> dict:
    steps = parse_proof_steps(rec["proof"])
    produced: dict[str, int] = {}
    for i, s in enumerate(steps):
        produced.setdefault(s["conclusion_id"], i)
    consumed = collections.Counter(p for s in steps for p in s["premises"])
    forward_ref = sum(1 for i, s in enumerate(steps) for p in s["premises"]
                      if p.startswith("int") and produced.get(p, 10**9) >= i)
    leaves = {p for s in steps for p in s["premises"] if p.startswith("sent")}
    other = [p for s in steps for p in s["premises"] if not re.fullmatch(r"(sent|int)\d+", p)]

    height: dict[str, int] = {}
    step_h = []
    for s in steps:
        h = 1 + max((height.get(p, 0) for p in s["premises"]), default=0)
        height[s["conclusion_id"]] = h
        step_h.append(h)
    root = steps[-1]["conclusion_id"] if steps else None
    depth = height.get(root, 0)

    # distance of each leaf from the root, in steps (root step = 1)
    dist: dict[str, int] = {}
    by_conc = {s["conclusion_id"]: s for s in steps}

    def down(node, d):
        s = by_conc.get(node)
        if s is None:
            return
        for p in s["premises"]:
            if p.startswith("sent"):
                dist[p] = min(dist.get(p, 99), d)
            else:
                down(p, d + 1)

    if root:
        down(root, 1)

    n_int_prem = [sum(p.startswith("int") for p in s["premises"]) for s in steps]
    kinds = ["leaf-only" if k == 0 else ("mixed" if k < len(s["premises"]) else "int-only")
             for k, s in zip(n_int_prem, steps)]
    n = len(steps)
    shape = "single-step" if n == 1 else ("branched" if any(k >= 2 for k in n_int_prem) else "linear")
    return {
        "n_steps": n,
        "fan_in": [len(s["premises"]) for s in steps],
        "step_kinds": kinds,
        "n_leaves": len(leaves),
        "depth": depth,
        "shape": shape,
        "join_steps": sum(k >= 2 for k in n_int_prem),
        "base_steps": sum(k == 0 for k in n_int_prem),
        "leaf_dist": list(dist.values()),
        "max_width": max(collections.Counter(step_h).values(), default=0),
        "reused_nodes": sum(1 for c in consumed.values() if c > 1),
        "forward_refs": forward_ref,
        "bad_tokens": len(other),
        "final_is_last_and_only": sum(s["conclusion_id"] == "hypothesis" for s in steps) == 1
                                  and bool(steps) and steps[-1]["conclusion_id"] == "hypothesis",
        "int_order_ascending": [s["conclusion_id"] for s in steps[:-1]] == [f"int{i}" for i in range(1, n)],
        "unused_triples": len(set(rec["meta"]["triples"]) - leaves),
        "int_defined_not_consumed": sum(1 for c in produced if c != "hypothesis" and c not in consumed),
        "int_premise_missing_def": sum(1 for p in consumed if p.startswith("int") and p not in produced),
    }


def graph_section(t1_rows: dict[str, list[dict]]) -> dict:
    allr = [(s, r) for s, rs in t1_rows.items() for r in rs]
    A = [(s, r, analyze_tree(r)) for s, r in allr]

    def per_split(fn):
        return {s: fn([a for ss, _, a in A if ss == s]) for s in SPLITS} | {"all": fn([a for _, _, a in A])}

    shape = per_split(lambda L: hist([a["shape"] for a in L]))
    out = {
        "n_trees": len(A),
        "depth": per_split(lambda L: {"summary": summarize([a["depth"] for a in L]), "hist": hist([a["depth"] for a in L])}),
        "n_steps": per_split(lambda L: {"summary": summarize([a["n_steps"] for a in L]), "hist": hist([a["n_steps"] for a in L])}),
        "n_leaves": per_split(lambda L: {"summary": summarize([a["n_leaves"] for a in L]), "hist": hist([a["n_leaves"] for a in L])}),
        "shape": shape,
        "fan_in_per_step": {"summary": summarize([f for _, _, a in A for f in a["fan_in"]]),
                            "hist": hist([f for _, _, a in A for f in a["fan_in"]])},
        "step_kinds": hist([k for _, _, a in A for k in a["step_kinds"]]),
        "join_steps_per_tree": hist([a["join_steps"] for _, _, a in A]),
        "base_steps_per_tree": hist([a["base_steps"] for _, _, a in A]),
        "max_width_per_tree": hist([a["max_width"] for _, _, a in A]),
        "leaf_distance_to_root": {"summary": summarize([d for _, _, a in A for d in a["leaf_dist"]]),
                                  "hist": hist([d for _, _, a in A for d in a["leaf_dist"]])},
        "mean_fan_in_by_shape": {sh: round(st.mean([f for _, _, a in A if a["shape"] == sh for f in a["fan_in"]]), 3)
                                 for sh in ("single-step", "linear", "branched")},
        "steps_by_shape": {sh: summarize([a["n_steps"] for _, _, a in A if a["shape"] == sh])
                           for sh in ("single-step", "linear", "branched")},
        "depth_by_shape": {sh: summarize([a["depth"] for _, _, a in A if a["shape"] == sh])
                           for sh in ("single-step", "linear", "branched")},
        "depth_x_steps": {str(d): hist([a["n_steps"] for _, _, a in A if a["depth"] == d])
                          for d in sorted({a["depth"] for _, _, a in A})},
        "checks": {
            "trees_with_reused_node(DAG not tree)": sum(1 for _, _, a in A if a["reused_nodes"]),
            "trees_with_forward_reference": sum(1 for _, _, a in A if a["forward_refs"]),
            "trees_with_nonstandard_token": sum(1 for _, _, a in A if a["bad_tokens"]),
            "trees_final_step_not_unique_last_hypothesis": sum(1 for _, _, a in A if not a["final_is_last_and_only"]),
            "trees_intermediates_not_numbered_1..n-1_in_order": sum(1 for _, _, a in A if not a["int_order_ascending"]),
            "trees_with_unused_triple": sum(1 for _, _, a in A if a["unused_triples"]),
            "trees_with_dangling_intermediate": sum(1 for _, _, a in A if a["int_defined_not_consumed"]),
            "trees_with_undefined_intermediate_premise": sum(1 for _, _, a in A if a["int_premise_missing_def"]),
            "depth_of_proof_field == recomputed depth": sum(1 for _, r, a in A if r["depth_of_proof"] == a["depth"]),
            "depth_of_proof_field > recomputed depth": sum(1 for _, r, a in A if r["depth_of_proof"] > a["depth"]),
            "depth_of_proof_field < recomputed depth": sum(1 for _, r, a in A if r["depth_of_proof"] < a["depth"]),
            "length_of_proof_field == n_steps": sum(1 for _, r, a in A if r["length_of_proof"] == a["n_steps"]),
            "length_of_proof_field < depth_of_proof_field": sum(1 for _, r, _a in A if r["length_of_proof"] < r["depth_of_proof"]),
        },
        "depth_field_vs_recomputed_examples": [
            {"id": r["id"], "split": s, "depth_of_proof": r["depth_of_proof"], "recomputed_depth": a["depth"],
             "length_of_proof": r["length_of_proof"]}
            for s, r, a in A if r["depth_of_proof"] != a["depth"]][:8],
    }
    return out


# --------------------------------------------------------------------------- #
def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def integrity_section(raw: dict) -> dict:
    t1 = {s: raw[(1, s)] for s in SPLITS}
    out: dict = {}
    allr = [r for s in SPLITS for r in t1[s]]

    # duplicate ids
    c = collections.Counter(r["id"] for r in allr)
    dup = {i: n for i, n in c.items() if n > 1}
    out["duplicate_ids"] = {
        "n_ids": len(dup),
        "records_affected": sum(dup.values()),
        "ids": {i: {"split": [s for s in SPLITS for r in t1[s] if r["id"] == i],
                    "hypotheses_differ": len({r["hypothesis"] for r in allr if r["id"] == i}) > 1}
                for i in dup},
    }

    # task-2 / task-3 row alignment with task-1
    for t in (2, 3):
        aligned = sum(1 for s in SPLITS for a, b in zip(t1[s], raw[(t, s)])
                      if a["id"] == b["id"] and a["hypothesis"] == b["hypothesis"] and a["question"] == b["question"])
        out[f"task{t}_row_aligned_with_task1(id+hypothesis+question)"] = f"{aligned}/{len(allr)}"

    # task 1: context vs triples vs provenance
    out["task1_context_equals_triples"] = sum(parse_context(r["context"]) == r["meta"]["triples"] for r in allr)
    out["task1_context_keys_are_sent1..n"] = sum(
        list(r["meta"]["triples"]) == [f"sent{i}" for i in range(1, len(r["meta"]["triples"]) + 1)] for r in allr)
    out["task1_proof_equals_step_proof"] = sum(r["proof"] == r["meta"]["step_proof"] for r in allr)
    out["task1_hypothesis_text_equals_intermediate_conclusions[hypothesis_id]"] = sum(
        r["hypothesis"] == r["meta"]["intermediate_conclusions"].get(r["meta"]["hypothesis_id"]) for r in allr)
    out["task1_hypothesis_matches_that_after_norm"] = sum(
        norm(r["hypothesis"]) == norm(r["meta"]["intermediate_conclusions"].get(r["meta"]["hypothesis_id"], "")) for r in allr)
    out["task1_hypothesis_id==int{length_of_proof}"] = sum(
        r["meta"]["hypothesis_id"] == f"int{r['length_of_proof']}" for r in allr)
    out["task1_n_intermediate_conclusions==length_of_proof"] = sum(
        len(r["meta"]["intermediate_conclusions"]) == r["length_of_proof"] for r in allr)
    # inline text in proof vs intermediate_conclusions
    mism = tot = 0
    for r in allr:
        for s in parse_proof_steps(r["proof"]):
            if s["conclusion_text"] is not None:
                tot += 1
                mism += s["conclusion_text"] != r["meta"]["intermediate_conclusions"].get(s["conclusion_id"])
    out["task1_inline_int_text_vs_meta"] = {"inline_conclusions": tot, "mismatches": mism}
    out["task1_final_step_has_inline_text"] = sum(
        parse_proof_steps(r["proof"])[-1]["conclusion_text"] is not None for r in allr)
    pv_ok = 0
    n_leaf = 0
    for r in allr:
        pt = {v["original_text"] for v in r["meta"]["worldtree_provenance"].values()}
        for v in r["meta"]["triples"].values():
            n_leaf += 1
            pv_ok += v in pt
    out["task1_leaf_text_in_own_provenance_original_text"] = f"{pv_ok}/{n_leaf}"
    out["task1_provenance_keys_equal_triple_keys"] = sum(
        set(r["meta"]["worldtree_provenance"]) == set(r["meta"]["triples"]) for r in allr)
    out["task1_add_list_sids_subset_of_triples"] = sum(
        all(a["sid"] in r["meta"]["triples"] for a in r["meta"]["add_list"]) for r in allr)

    # task 2
    ok_ctx = ok_gold = 0
    pos = []
    t2_leaf_text_equal = 0
    for s in SPLITS:
        for a, b in zip(t1[s], raw[(2, s)]):
            ctx = parse_context(b["context"])
            ok_ctx += len(ctx) == 25
            gold_ids = {p for st_ in parse_proof_steps(b["proof"]) for p in st_["premises"] if p.startswith("sent")}
            ok_gold += gold_ids <= set(ctx) and not (gold_ids & set(b["meta"]["distractors"]))
            pos += [int(g[4:]) for g in gold_ids]
            t2_leaf_text_equal += {ctx[g] for g in gold_ids} == set(a["meta"]["triples"].values())
    out["task2_context_has_25_sentences"] = ok_ctx
    out["task2_gold_ids_in_context_and_not_distractor"] = ok_gold
    out["task2_gold_leaf_text_set == task1_leaf_text_set"] = t2_leaf_text_equal
    out["task2_triples_equals_context"] = sum(
        parse_context(b["context"]) == b["meta"]["triples"] for s in SPLITS for b in raw[(2, s)])
    out["task2_proof_leaf_ids_renumbered_vs_task1"] = sum(
        a["proof"] != b["proof"] for s in SPLITS for a, b in zip(t1[s], raw[(2, s)]))
    out["task2_meta.triples_size"] = hist(len(b["meta"]["triples"]) for s in SPLITS for b in raw[(2, s)])
    out["task2_distractor_count"] = summarize([len(b["meta"]["distractors"]) for s in SPLITS for b in raw[(2, s)]])
    out["task2_gold_sentence_position_in_context"] = summarize(pos)

    # task 3
    exact = norm_hit = total = 0
    any_missing = 0
    cover = []
    dup_in_ctx = 0
    for s in SPLITS:
        for a, b in zip(t1[s], raw[(3, s)]):
            ctx = list(parse_context(b["context"]).values())
            cs, cn = set(ctx), {norm(x) for x in ctx}
            leaves = list(a["meta"]["triples"].values())
            e = sum(l in cs for l in leaves)
            n_ = sum(norm(l) in cn for l in leaves)
            exact += e
            norm_hit += n_
            total += len(leaves)
            any_missing += n_ < len(leaves)
            cover.append(n_ / len(leaves))
            dup_in_ctx += len(ctx) != len(cs)
    out["task3_gold_leaves_found_in_context_exact"] = f"{exact}/{total}"
    out["task3_gold_leaves_found_in_context_after_normalisation"] = f"{norm_hit}/{total}"
    out["task3_trees_with_at_least_one_gold_leaf_missing_from_context"] = f"{any_missing}/{len(cover)}"
    out["task3_per_tree_gold_coverage"] = summarize(cover)
    out["task3_contexts_with_duplicate_sentences"] = f"{dup_in_ctx}/{len(cover)}"
    out["task3_context_equals_meta.triples"] = sum(
        parse_context(b["context"]) == b["meta"]["triples"] for s in SPLITS for b in raw[(3, s)])
    out["task3_all.jsonl_equals_concat_of_splits"] = (
        raw[(3, "all")] == raw[(3, "train")] + raw[(3, "dev")] + raw[(3, "test")]
        or sorted(r["id"] for r in raw[(3, "all")]) == sorted(r["id"] for s in SPLITS for r in raw[(3, s)]))
    return out


# --------------------------------------------------------------------------- #
def pipeline_section(data: Path) -> dict:
    out: dict = {"per_task_split": {}}
    ex_keys, meta_keys = set(), set()
    dep_none = 0
    unresolved = 0
    borrowed_wrong = 0
    for t in TASKS:
        for s in SPLITS:
            ds = EntailmentBank(task=t, split=s, local_dir=str(data), shuffle_seed=None)
            exs = list(ds)
            for e in exs:
                ex_keys |= set(e.__dict__)
                meta_keys |= set(e.meta)
                dep_none += e.meta["depth_of_proof"] is None
                unresolved += sum(1 for st_ in e.meta["entailment_steps"]
                                  for p, pid in zip(st_["premises"], st_["premise_ids"]) if p == pid)
            n_steps = [len(e.meta["entailment_steps"]) for e in exs]
            out["per_task_split"][f"task{t}/{s}"] = {
                "examples": len(exs), "unique_ids": len({e.id for e in exs}),
                "steps": summarize(n_steps),
                "prompt_chars": summarize([len(e.prompt) for e in exs]),
                "reasoning_chars": summarize([len(e.meta["reasoning"]) for e in exs]),
                "borrowed_proof": sum(e.meta["proof_borrowed_from_task1"] for e in exs),
                "n_distractors": summarize([e.meta["n_distractors"] for e in exs]),
            }
            if t == 3:
                gold = EntailmentBank(task=1, split=s, local_dir=str(data), shuffle_seed=None)._records
                # a borrowed tree should carry the record's own hypothesis
                raw3 = ds._records
                borrowed_wrong += sum(1 for r in raw3
                                      if ds._gold_trees[str(r["id"])]["hypothesis"] != r["hypothesis"])
                del gold
    out["example_fields"] = sorted(ex_keys)
    out["example_meta_keys"] = sorted(meta_keys)
    out["examples_with_depth_of_proof_None"] = dep_none
    out["steps_whose_premise_text_is_unresolved_id"] = unresolved
    out["task3_records_borrowing_a_task1_tree_with_different_hypothesis"] = borrowed_wrong
    # step fields
    ex = next(iter(EntailmentBank(task=1, split="dev", local_dir=str(data), shuffle_seed=None)))
    out["example_instance"] = {k: v for k, v in ex.__dict__.items() if k != "meta"}
    out["example_instance_meta"] = {k: v for k, v in ex.meta.items() if k not in ("entailment_steps",)}
    out["example_instance_meta"]["entailment_steps"] = ex.meta["entailment_steps"][:2]
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/entailment_trees_emnlp2021_data_v3"))
    ap.add_argument("--out", default=str(ROOT / "docs/schemas/entailmentbank_stats.json"))
    a = ap.parse_args()
    data = Path(a.data)
    if not (data / "dataset").is_dir():
        sys.exit(f"{data}/dataset not found; pass --data")

    raw = {}
    for t in TASKS:
        for s in SPLITS:
            raw[(t, s)] = read_jsonl(data / "dataset" / f"task_{t}" / f"{s}.jsonl")
    raw[(3, "all")] = read_jsonl(data / "dataset/task_3/all.jsonl")
    corpus = json.loads((data / "supporting_data/worldtree_corpus_sentences_extended.json").read_text(encoding="utf-8"))

    stats = {
        "counts": {f"task{t}/{s}": len(raw[(t, s)]) for (t, s) in raw},
        "fields": {f"task{t}": infer_fields([r for s in SPLITS for r in raw[(t, s)]]) for t in TASKS},
        "graph": graph_section({s: raw[(1, s)] for s in SPLITS}),
        "integrity": integrity_section(raw),
        "pipeline": pipeline_section(data),
        "supporting_corpus": {
            "entries": len(corpus),
            "key_forms": {"uuid-like": sum(1 for k in corpus if re.fullmatch(r"[0-9a-f]{4}(-[0-9a-f]{4}){3}", k)),
                          "ET_addon:N": sum(1 for k in corpus if k.startswith("ET_addon:"))},
            "words_per_sentence": summarize([len(v.split()) for v in corpus.values()]),
            "duplicate_sentences": len(corpus) - len(set(corpus.values())),
            "placeholder_entries": [k for k, v in corpus.items() if v == "DUMMY"],
            "provenance_uuid_found_as_key": sum(
                1 for s in SPLITS for r in raw[(1, s)] for p in r["meta"]["worldtree_provenance"].values()
                if p["uuid"] in corpus),
            "provenance_entries": sum(len(r["meta"]["worldtree_provenance"]) for s in SPLITS for r in raw[(1, s)]),
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
