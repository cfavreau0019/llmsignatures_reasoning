import json
from pathlib import Path

import pytest

from reasoning_trajectories.config import Config
from reasoning_trajectories.datasets import load_dataset_from_config
from reasoning_trajectories.datasets.entailmentbank import (
    EntailmentBank,
    build_id2text,
    facts_block,
    linearize_proof,
    parse_context,
    parse_proof_steps,
)
from reasoning_trajectories.reasoning import ReasoningProvider

SAMPLE = Path(__file__).parent / "data" / "entailmentbank_task_1_dev.jsonl"
RECORDS = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_parse_context():
    m = parse_context("sent1: a magnet attracts metals. sent2: iron is a metal.")
    assert m == {"sent1": "a magnet attracts metals.", "sent2": "iron is a metal."}


def test_build_id2text_prefers_triples_and_adds_hypothesis():
    m = build_id2text(RECORDS[0])
    assert m["sent2"].startswith("winter is when")
    assert m["int1"].startswith("in december")
    assert m["hypothesis"].startswith("in december the northern hemisphere")


def test_build_id2text_falls_back_to_context_when_no_triples():
    rec = RECORDS[1]
    assert "triples" not in (rec.get("meta") or {})
    m = build_id2text(rec)
    assert m["sent3"] == "a nail is made of iron."
    assert m["int1"] == "a nail is made of a magnetic metal"


def test_parse_proof_steps_handles_prefix_and_intermediate_text():
    steps = parse_proof_steps(RECORDS[1]["proof"])  # has "$proof$ = " prefix
    assert steps[0] == {
        "premises": ["sent2", "sent3"],
        "conclusion_id": "int1",
        "conclusion_text": "a nail is made of a magnetic metal",
    }
    assert steps[1]["conclusion_id"] == "hypothesis"
    assert steps[1]["premises"] == ["sent1", "int1"]


def test_linearize_natural_resolves_all_ids():
    text, structured = linearize_proof(RECORDS[0], fmt="natural")
    assert "sent1" not in text and "int1" not in text
    assert text.splitlines()[0].startswith("1. Since ")
    assert len(structured) == 2
    assert structured[-1]["conclusion"].startswith("in december the northern hemisphere")


def test_linearize_structured_and_raw():
    s, _ = linearize_proof(RECORDS[0], fmt="structured")
    assert "=>" in s and "Step 1:" in s
    r, _ = linearize_proof(RECORDS[1], fmt="raw")
    assert "sent" not in r and "int1" not in r
    assert "&" in r  # keeps proof connectives, ids expanded


def test_facts_block_numbered_in_order():
    fb = facts_block(build_id2text(RECORDS[1]))
    lines = fb.splitlines()
    assert lines[0].startswith("1. ") and lines[3].startswith("4. ")
    assert "int" not in fb


def test_loader_yields_examples_with_gold_reasoning(tmp_path):
    # official layout: <root>/dataset/task_1/dev.jsonl
    d = tmp_path / "dataset" / "task_1"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    ds = EntailmentBank(task=1, split="validation", local_dir=tmp_path, shuffle_seed=None)
    exs = list(ds)
    assert len(exs) == 2
    ex = exs[0]
    assert ex.id == "EB_TEST_001"
    assert ex.prompt.rstrip().endswith("Reasoning:")
    assert "1. the northern hemisphere is a kind of place" in ex.prompt
    assert ex.meta["reasoning"].startswith("1. Since ")
    assert ex.meta["length_of_proof"] == 2
    assert ex.meta["n_distractors"] == 0


def test_reasoning_provider_uses_dataset_field(tmp_path):
    d = tmp_path / "task_1"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    ds = EntailmentBank(task=1, split="dev", local_dir=tmp_path, shuffle_seed=None)
    ex = next(iter(ds))

    cfg = Config.load("configs/default.yaml",
                      overrides={"reasoning.source": "dataset_field", "reasoning.stop_on_answer": False})
    provider = ReasoningProvider(cfg.reasoning, model=None)
    assert provider.reasoning_for(ex) == ex.meta["reasoning"]


def test_load_from_config(tmp_path):
    d = tmp_path / "dataset" / "task_2"
    d.mkdir(parents=True)
    (d / "test.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = Config.load("configs/default.yaml", overrides={
        "dataset.name": "entailmentbank",
        "dataset.split": "test",
        "dataset.params": {"task": 2, "local_dir": str(tmp_path), "reasoning_format": "structured"},
    })
    ds = load_dataset_from_config(cfg.dataset)
    ex = next(iter(ds))
    assert ex.meta["task"] == 2
    assert "Step 1:" in ex.meta["reasoning"]


def test_missing_source_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so the conventional data/raw/... path can't resolve
    with pytest.raises(ValueError, match="local_dir"):
        EntailmentBank(task=1, split="dev")
