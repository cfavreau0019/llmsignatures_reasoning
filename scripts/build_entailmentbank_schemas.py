"""Write + validate JSON Schemas for EntailmentBank v3 and the repo objects built from it.

    python scripts/build_entailmentbank_schemas.py --data data/raw/entailment_trees_emnlp2021_data_v3

Writes ``docs/schemas/*.schema.json`` (JSON Schema draft 2020-12), then validates
*every* raw record, every ``Example`` the loader yields, every ``ReasoningChain``
(valid + corrupted) against them. Exits non-zero on any violation, so the schemas
cannot silently drift from the data or the dataclasses.

JSON Schema cannot express cross-field equalities (e.g. ``context`` == ``meta.triples``);
those live in the ``x-cross-field-invariants`` annotation on each root schema and are
checked in ``scripts/profile_entailmentbank.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_trajectories.datasets import EntailmentBank, PairedCorruption  # noqa: E402
from reasoning_trajectories.reasoning.chain import CORRUPTIONS, ReasoningChain  # noqa: E402

DRAFT = "https://json-schema.org/draft/2020-12/schema"
OUT = ROOT / "docs/schemas"
SENT = r"^sent[0-9]+$"
INT = r"^int[0-9]+$"
# "sent1 & sent2 -> int1: text; int1 & sent3 -> hypothesis; "
PROOF = (r"^( *(sent|int)[0-9]+( & (sent|int)[0-9]+)* -> (int[0-9]+: [^;]+|hypothesis);)+ *$")
CONTEXT = r"^sent[0-9]+: "


def s(**kw):
    return kw


STR = {"type": "string"}
NONEMPTY = {"type": "string", "minLength": 1}


# --------------------------------------------------------------------------- #
# raw records
# --------------------------------------------------------------------------- #
def sentence_map(desc: str, key=SENT):
    return {"type": "object", "description": desc, "patternProperties": {key: NONEMPTY},
            "additionalProperties": False}


def raw_schema(task: int) -> dict:
    meta_props: dict = {
        "question_text": {**STR, "description": "Duplicate of top-level `question`."},
        "answer_text": {**STR, "description": "Duplicate of top-level `answer`."},
        "triples": sentence_map(
            {1: "Gold leaf sentences only (sent1..sentN); equals the parsed `context`.",
             2: "All 25 context sentences (gold + distractors); equals the parsed `context`.",
             3: "All 25 retrieved corpus sentences; equals the parsed `context`. Not all gold leaves are present."}[task]),
        "core_concepts": {"type": "array", "items": NONEMPTY,
                          "description": "Annotator-marked key statements. Empty for 3 records."},
    }
    required_meta = ["question_text", "answer_text", "triples", "core_concepts", "worldtree_provenance"]
    if task in (1, 2):
        meta_props["worldtree_provenance"] = {
            "type": "object",
            "description": "Gold leaves only (even in task 2). `uuid` is '' for annotator-added facts "
                           "(== the `add_list` entries).",
            "patternProperties": {SENT: {"type": "object", "required": ["uuid", "original_text"],
                                         "properties": {"uuid": STR, "original_text": NONEMPTY},
                                         "additionalProperties": False}},
            "additionalProperties": False}
        meta_props.update({
            "hypothesis_id": {"type": "string", "pattern": INT,
                              "description": "Id of the root, always `int{length_of_proof}`."},
            "intermediate_conclusions": sentence_map(
                "One entry per proof step (including the root `hypothesis_id`).", INT),
            "distractors": {"type": "array", "items": {"type": "string", "pattern": SENT},
                            **({"maxItems": 0, "description": "Always empty in task 1."} if task == 1 else
                               {"minItems": 8, "description": "Ids in `triples` that are NOT gold leaves."})},
            "distractors_relevance": {"type": "array", "items": {"type": "number"},
                                      **({"maxItems": 0} if task == 1 else {"minItems": 8}),
                                      "description": "Retrieval score per distractor, parallel to `distractors`."},
            "step_proof": {**STR, "description": "Identical to top-level `proof`."},
            "lisp_proof": {**STR, "description": "S-expression encoding; the root is `intN`, not `hypothesis`."},
            "polish_proof": {**STR, "description": "Prefix-notation encoding."},
            "add_list": {"type": "array", "description": "Facts the annotators wrote (not in WorldTree).",
                         "items": {"type": "object", "required": ["sid", "fact"],
                                   "properties": {"sid": {"type": "string", "pattern": SENT}, "fact": NONEMPTY},
                                   "additionalProperties": False}},
            "delete_list": {"type": "array", "description": "WorldTree facts shown to annotators but unused.",
                            "items": {"type": "object", "required": ["uuid", "fact"],
                                      "properties": {"uuid": NONEMPTY, "fact": NONEMPTY},
                                      "additionalProperties": False}},
        })
        required_meta += ["hypothesis_id", "intermediate_conclusions", "distractors", "distractors_relevance",
                          "step_proof", "lisp_proof", "polish_proof", "add_list", "delete_list"]
    else:
        meta_props["worldtree_provenance"] = {
            "type": "array",
            "description": "ARRAY (not an object as in tasks 1/2): one entry per context sentence.",
            "items": {"type": "object", "required": ["sid", "uuid", "relevance_score"],
                      "properties": {"sid": {"type": "string", "pattern": SENT}, "uuid": STR,
                                     "relevance_score": {"type": "number"}},
                      "additionalProperties": False}}

    props = {
        "id": {**NONEMPTY, "description": "Source-question id. NOT unique: 6 ids occur twice (see docs)."},
        "context": {"type": "string", "pattern": CONTEXT,
                    "description": "`sent1: <text> sent2: <text> ...` (single string, no newlines)."},
        "question": NONEMPTY,
        "answer": NONEMPTY,
        "hypothesis": {**NONEMPTY, "description": "Declarative restatement of question+answer; the proof root."},
        "proof": ({"type": "string", "const": "", "description": "Always '' in task 3 (retrieval only)."}
                  if task == 3 else
                  {"type": "string", "pattern": PROOF,
                   "description": "Steps separated by ';'. Intermediate conclusions carry inline text; "
                                  "the final step ends in the literal `hypothesis`."}),
        "meta": {"type": "object", "required": required_meta, "properties": meta_props,
                 "additionalProperties": False},
    }
    required = ["id", "context", "question", "answer", "hypothesis", "proof", "meta"]
    if task in (1, 2):
        props.update({
            "full_text_proof": {**STR, "description": "Proof with every sentN replaced by its text; "
                                                       "` [BECAUSE] a [AND] b [INFER] intN: t ...`. Final step ends "
                                                       "in `intN`, not `hypothesis`."},
            "depth_of_proof": {"type": "integer", "minimum": 0,
                               "description": "Annotator-reported depth. Disagrees with the depth recomputed from "
                                              "`proof` for 46/1840 records."},
            "length_of_proof": {"type": "integer", "minimum": 1,
                                "description": "Number of steps; always equals the number of ';'-separated steps."},
        })
        required += ["full_text_proof", "depth_of_proof", "length_of_proof"]

    invariants = {
        1: ["parse_context(context) == meta.triples",
            "proof == meta.step_proof",
            "meta.hypothesis_id == 'int' + str(length_of_proof)",
            "meta.intermediate_conclusions[meta.hypothesis_id] == hypothesis",
            "len(meta.intermediate_conclusions) == length_of_proof",
            "every inline `intN: text` in proof == meta.intermediate_conclusions[intN]",
            "meta.worldtree_provenance.keys() == meta.triples.keys()",
            "every meta.add_list[].sid in meta.triples",
            "every sentN in proof in meta.triples; every key of meta.triples used in proof"],
        2: ["parse_context(context) == meta.triples (25 entries)",
            "len(meta.triples) == len(gold leaves) + len(meta.distractors)",
            "proof leaf ids subset of context ids and disjoint from meta.distractors",
            "gold leaf text set == the task-1 record's meta.triples text set (same row index)",
            "meta.worldtree_provenance.keys() == gold leaf ids only",
            "proof == meta.step_proof; meta.hypothesis_id == 'int' + str(length_of_proof)"],
        3: ["parse_context(context) == meta.triples (25 entries)",
            "row i has the same id/question/hypothesis as task-1 and task-2 row i",
            "gold leaves are only partly present in the context (mean coverage 0.77 after normalisation)",
            "len(meta.worldtree_provenance) == len(meta.triples)"],
    }[task]
    return {"$schema": DRAFT, "$id": f"entailmentbank-v3/task_{task}.record.schema.json",
            "title": f"EntailmentBank v3 task_{task} record (one JSONL line)",
            "type": "object", "required": required, "properties": props, "additionalProperties": False,
            "x-cross-field-invariants": invariants}


def corpus_schema() -> dict:
    return {"$schema": DRAFT, "$id": "entailmentbank-v3/worldtree_corpus.schema.json",
            "title": "supporting_data/worldtree_corpus_sentences_extended.json",
            "type": "object",
            "description": "Single JSON object: WorldTree fact uuid (or `ET_addon:N` for EntailmentBank additions) -> sentence. "
                           "11,941 entries; 420 duplicate sentences; `ET_addon:0` is the placeholder 'DUMMY'.",
            "patternProperties": {r"^([0-9a-f]{4}-){3}[0-9a-f]{4}$|^ET_addon:[0-9]+$": NONEMPTY},
            "additionalProperties": False}


# --------------------------------------------------------------------------- #
# repo objects
# --------------------------------------------------------------------------- #
STEP_ID = {"type": "string", "pattern": r"^(sent[0-9]+|int[0-9]+|SUB)$"}
CONC_ID = {"type": "string", "pattern": r"^(int[0-9]+|hypothesis|SUB)?$"}


def chain_schema() -> dict:
    corruption_kinds = list(CORRUPTIONS)
    return {
        "$schema": DRAFT, "$id": "reasoning_trajectories/reasoning_chain.schema.json",
        "title": "ReasoningChain (dataclasses.asdict) and its serialised steps",
        "type": "object",
        "required": ["steps", "fmt", "source_id", "meta", "raw_text"],
        "properties": {
            "steps": {"type": "array", "items": {"$ref": "#/$defs/Step"},
                      "description": "[] for opaque `from_text` chains."},
            "fmt": {"enum": ["natural", "structured", "raw", "text"]},
            "source_id": {"type": ["string", "null"]},
            "meta": {"type": "object",
                     "properties": {"condition": {"enum": ["valid", "corrupted"]},
                                    "pair_id": {"type": "string"},
                                    "corruption": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/Corruption"}],
                                                   "description": "null on the `valid` member of a pair."}},
                     "additionalProperties": False},
            "raw_text": {"type": ["string", "null"], "description": "Set only for opaque chains."},
        },
        "additionalProperties": False,
        "$defs": {
            "Step": {
                "title": "Step (frozen dataclass)", "type": "object",
                "required": ["premises", "conclusion", "premise_ids", "conclusion_id"],
                "properties": {
                    "premises": {"type": "array", "minItems": 1, "items": NONEMPTY},
                    "conclusion": NONEMPTY,
                    "premise_ids": {"type": "array", "items": STEP_ID,
                                    "description": "Parallel to `premises`; may be empty. `SUB` marks a corrupted slot."},
                    "conclusion_id": {**CONC_ID, "description": "`intN`, `hypothesis` (final step), `SUB` (corrupted) or ''."},
                }, "additionalProperties": False},
            "StepDict": {
                "title": "Step as serialised by ReasoningChain.to_dicts() / Example.meta['entailment_steps']",
                "type": "object",
                "required": ["step", "premise_ids", "premises", "conclusion_id", "conclusion"],
                "properties": {"step": {"type": "integer", "minimum": 1, "description": "1-based position."},
                               "premise_ids": {"type": "array", "items": STEP_ID},
                               "premises": {"type": "array", "minItems": 1, "items": NONEMPTY},
                               "conclusion_id": CONC_ID, "conclusion": NONEMPTY},
                "additionalProperties": False},
            "Corruption": {
                "title": "ReasoningChain.meta['corruption'] change record",
                "type": "object", "required": ["kind"],
                "properties": {"kind": {"enum": corruption_kinds},
                               "skipped": {**STR, "description": "Present when the corruption was a no-op."},
                               "permutation": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                               "step": {"type": "integer", "minimum": 1},
                               "premise_index": {"type": "integer", "minimum": 0},
                               "old": STR, "new": STR,
                               "propagated_to": {"type": "integer", "minimum": 0}},
                "additionalProperties": False},
        },
        "x-cross-field-invariants": [
            "raw_text is not None  <=>  fmt == 'text'  <=>  steps == []",
            "at most the final step has conclusion_id == 'hypothesis' (before corruption)",
            "len(premise_ids) in (0, len(premises))",
            "a corrupted chain has meta.condition == 'corrupted' and meta.corruption.kind in CORRUPTIONS",
        ],
    }


def example_schema() -> dict:
    return {
        "$schema": DRAFT, "$id": "reasoning_trajectories/example.schema.json",
        "title": "Example (dataclasses.asdict) as yielded by EntailmentBank / PairedCorruption",
        "type": "object",
        "required": ["id", "question", "choices", "choice_labels", "answer_key", "prompt", "context", "meta"],
        "properties": {
            "id": {**NONEMPTY, "description": "Raw `id`. Paired corrupted variants append `__<kind>`."},
            "question": {**NONEMPTY, "description": "Raw `question`, or the hypothesis if the question is empty."},
            "choices": {"type": "array", "items": NONEMPTY, "maxItems": 1,
                        "description": "EntailmentBank: `[answer]` (a single short answer, not a multiple-choice list)."},
            "choice_labels": {"type": "array", "items": {"const": "A"}, "maxItems": 1},
            "answer_key": {"enum": ["A", None]},
            "prompt": {**NONEMPTY, "description": "Model input up to 'Reasoning:'; embeds the numbered fact list."},
            "context": {"type": ["string", "null"], "description": "Newline-joined `N. <sentence>` fact list."},
            "meta": {
                "type": "object",
                "required": ["dataset", "task", "split", "hypothesis", "answer_text", "reasoning",
                             "reasoning_format", "entailment_steps", "proof_raw", "proof_borrowed_from_task1",
                             "depth_of_proof", "length_of_proof", "n_distractors"],
                "properties": {
                    "dataset": {"const": "entailmentbank"},
                    "task": {"enum": [1, 2, 3]},
                    "split": {"enum": ["train", "dev", "test"]},
                    "hypothesis": NONEMPTY, "answer_text": NONEMPTY,
                    "reasoning": {**NONEMPTY, "description": "`ReasoningChain.text` in `reasoning_format`."},
                    "reasoning_format": {"enum": ["natural", "structured", "raw"]},
                    "entailment_steps": {"type": "array", "minItems": 1,
                                         "items": {"$ref": "reasoning_chain.schema.json#/$defs/StepDict"}},
                    "proof_raw": {**STR, "description": "Raw proof string (task-1 numbering for task 3)."},
                    "proof_borrowed_from_task1": {"type": "boolean", "description": "True for every task-3 example."},
                    "depth_of_proof": {"type": ["integer", "null"]},
                    "length_of_proof": {"type": "integer", "minimum": 1},
                    "n_distractors": {"type": "integer", "minimum": 0,
                                      "description": "0 for task 1 AND task 3 (task 3 does not mark distractors)."},
                    "condition": {"enum": ["valid", "corrupted"], "description": "Only after PairedCorruption."},
                    "pair_id": {"type": "string", "description": "Only after PairedCorruption."},
                    "corruption": {"anyOf": [{"type": "null"}, {"$ref": "reasoning_chain.schema.json#/$defs/Corruption"}],
                                   "description": "Only after PairedCorruption; null for the valid member."},
                },
                "additionalProperties": False},
        },
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
def make_validator(schema: dict, registry_schemas: list[dict]):
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
    reg = Registry().with_resources(
        [(sc["$id"], Resource.from_contents(sc, default_specification=DRAFT202012)) for sc in registry_schemas])
    return Draft202012Validator(schema, registry=reg)


def check(v: Draft202012Validator, inst, what: str, errors: list) -> None:
    for e in list(v.iter_errors(inst))[:1]:
        errors.append(f"{what}: {e.message[:200]} @ {list(e.absolute_path)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/entailment_trees_emnlp2021_data_v3"))
    a = ap.parse_args()
    data = Path(a.data)

    schemas = {f"entailmentbank_task{t}_record": raw_schema(t) for t in (1, 2, 3)}
    schemas["entailmentbank_worldtree_corpus"] = corpus_schema()
    schemas["reasoning_chain"] = chain_schema()
    schemas["example"] = example_schema()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sc in schemas.items():
        Draft202012Validator.check_schema(sc)
        (OUT / f"{name}.schema.json").write_text(json.dumps(sc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(schemas)} schemas to {OUT}")

    errors: list[str] = []
    counts: dict[str, int] = {}
    all_sc = list(schemas.values())

    # raw records
    for t in (1, 2, 3):
        v = make_validator(schemas[f"entailmentbank_task{t}_record"], all_sc)
        files = [data / "dataset" / f"task_{t}" / f"{sp}.jsonl" for sp in ("train", "dev", "test")]
        if t == 3:
            files.append(data / "dataset/task_3/all.jsonl")
        n = 0
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    n += 1
                    check(v, json.loads(line), f"{f.name} task{t}", errors)
        counts[f"raw task{t} records"] = n
    corpus = json.loads((data / "supporting_data/worldtree_corpus_sentences_extended.json").read_text(encoding="utf-8"))
    check(make_validator(schemas["entailmentbank_worldtree_corpus"], all_sc), corpus, "corpus", errors)
    counts["corpus entries"] = len(corpus)

    # Examples + chains
    ev = make_validator(schemas["example"], all_sc)
    cv = make_validator(schemas["reasoning_chain"], all_sc)
    n_ex = n_ch = n_corr = 0
    for t in (1, 2, 3):
        for sp in ("train", "dev", "test"):
            base = list(EntailmentBank(task=t, split=sp, local_dir=str(data), shuffle_seed=None))
            for ex in base:
                n_ex += 1
                check(ev, json.loads(json.dumps(dataclasses.asdict(ex))), f"Example t{t}/{sp}/{ex.id}", errors)
                ch = ReasoningChain.from_example(ex)
                n_ch += 1
                check(cv, json.loads(json.dumps(dataclasses.asdict(ch))), f"Chain t{t}/{sp}/{ex.id}", errors)
            if t == 1:                                          # corrupted variants
                for ex in PairedCorruption(base, min_steps=3):
                    n_ex += 1
                    check(ev, json.loads(json.dumps(dataclasses.asdict(ex))), f"Paired Example {ex.id}", errors)
                    ch = ReasoningChain.from_example(ex)
                    n_ch += 1
                    n_corr += ex.meta["condition"] == "corrupted"
                    check(cv, json.loads(json.dumps(dataclasses.asdict(ch))), f"Paired Chain {ex.id}", errors)
    ch = ReasoningChain.from_text("some generated prose", source_id="x")        # opaque chain
    check(cv, json.loads(json.dumps(dataclasses.asdict(ch))), "opaque chain", errors)
    counts.update({"Example instances": n_ex, "ReasoningChain instances": n_ch, "of which corrupted": n_corr})

    for k, v in counts.items():
        print(f"  validated {v:>6} {k}")
    if errors:
        print(f"\n{len(errors)} VIOLATIONS (first 15):")
        for e in errors[:15]:
            print("  ", e)
        sys.exit(1)
    print("all instances valid")


if __name__ == "__main__":
    main()
