import json
from pathlib import Path

import pytest

from reasoning_trajectories.datasets.entailmentbank import format_steps, resolve_steps
from reasoning_trajectories.reasoning import DonorPool, ReasoningChain, Step

SAMPLE = Path(__file__).parent / "data" / "entailmentbank_task_1_dev.jsonl"
RECORDS = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]
DEEP = RECORDS[4]          # EB_TEST_005, 4 steps, linear int1->int2->int3->hypothesis


# --------------------------------------------------------------------------- #
# construction + text
# --------------------------------------------------------------------------- #
def test_from_entailmentbank_text_matches_format_steps():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    assert ch.source_id == "EB_TEST_005"
    assert ch.text == format_steps(resolve_steps(DEEP), "natural")
    assert len(ch) == 4 and ch.is_structured


def test_from_steps_and_to_dicts_roundtrip():
    steps = resolve_steps(DEEP)
    ch = ReasoningChain.from_steps(steps, fmt="structured", source_id="x")
    assert ch.to_dicts() == steps
    assert "Step 1:" in ch.text


def test_as_format_returns_new_chain():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    raw = ch.as_format("raw")
    assert raw is not ch and ch.fmt == "natural" and raw.fmt == "raw"
    assert raw.text.endswith(";") and "Since" not in raw.text


def test_components():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    assert ch.hypothesis == ch.conclusions[-1]
    assert len(ch.premises) == sum(len(s.premises) for s in ch.steps)
    assert all(p in ch.premises for p in ch.leaves)
    assert ch.conclusions[0] in ch.intermediates          # int1 is not the hypothesis


def test_from_text_is_opaque():
    ch = ReasoningChain.from_text("the cat sat on the mat", source_id="t")
    assert ch.text == "the cat sat on the mat"
    assert not ch.is_structured and len(ch) == 0
    bad = ch.corrupt("step_shuffle")
    assert bad.text == ch.text and bad.meta["corruption"]["skipped"]


# --------------------------------------------------------------------------- #
# corruption
# --------------------------------------------------------------------------- #
def test_corrupt_returns_chain_and_leaves_original_untouched():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    before = ch.text
    bad = ch.corrupt("step_shuffle", seed=0)
    assert isinstance(bad, ReasoningChain)
    assert bad.meta["corruption"]["kind"] == "step_shuffle"
    assert bad.text != before
    assert ch.text == before                              # immutable
    assert sorted(bad.conclusions) == sorted(ch.conclusions)


def test_shuffled_is_non_identity():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    perm = ch.shuffled(seed=1).meta["corruption"]["permutation"]
    assert perm != sorted(perm)


def test_short_chain_shuffle_escalates_to_premise_sub():
    ch = ReasoningChain.from_entailmentbank(RECORDS[0])   # 2 steps
    bad = ch.corrupt("step_shuffle", seed=0, min_steps=3)
    assert bad.meta["corruption"]["kind"] == "premise_substitution"


def test_premise_substitution_other_tree_needs_donors_and_injects_foreign():
    ch = ReasoningChain.from_entailmentbank(RECORDS[2])
    with pytest.raises(ValueError):
        ch.corrupt("premise_substitution_other_tree", seed=0)          # no donors
    pool = DonorPool(ReasoningChain.from_entailmentbank(r) for r in RECORDS)
    bad = ch.corrupt("premise_substitution_other_tree", seed=0, donors=pool)
    info = bad.meta["corruption"]
    assert info["new"] not in ch.premises and info["new"] not in ch.conclusions
    assert info["new"] in bad.premises


def test_conclusion_substitution_propagates():
    ch = ReasoningChain.from_entailmentbank(DEEP)
    pool = DonorPool(ReasoningChain.from_entailmentbank(r) for r in RECORDS)
    bad = ch.corrupt("conclusion_substitution", seed=0, donors=pool)
    info = bad.meta["corruption"]
    si = info["step"] - 1
    changed = [i for i, (a, b) in enumerate(zip(ch.steps, bad.steps)) if a.conclusion != b.conclusion]
    assert changed == [si]
    assert bad.steps[si].conclusion_id == "SUB"
    assert info["propagated_to"] >= 1 and info["new"] in bad.steps[si + 1].premises


def test_corrupt_is_deterministic_per_seed():
    a = ReasoningChain.from_entailmentbank(DEEP).corrupt("step_shuffle", seed=0)
    b = ReasoningChain.from_entailmentbank(DEEP).corrupt("step_shuffle", seed=0)
    c = ReasoningChain.from_entailmentbank(DEEP).corrupt("step_shuffle", seed=1)
    assert a.text == b.text and a.text != c.text


def test_unknown_corruption_raises():
    with pytest.raises(ValueError, match="unknown corruption"):
        ReasoningChain.from_entailmentbank(DEEP).corrupt("nonsense")


# --------------------------------------------------------------------------- #
# donor pool accepts chains or examples
# --------------------------------------------------------------------------- #
def test_donorpool_from_examples_and_chains_equivalent(tmp_path):
    from reasoning_trajectories.datasets.entailmentbank import EntailmentBank

    d = tmp_path / "dataset" / "task_1"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    exs = list(EntailmentBank(task=1, split="validation", local_dir=tmp_path, shuffle_seed=None))

    from_ex = DonorPool(exs)
    from_ch = DonorPool(ReasoningChain.from_example(e) for e in exs)
    assert sorted(t for t, _ in from_ex._prem) == sorted(t for t, _ in from_ch._prem)


# --------------------------------------------------------------------------- #
# provider integration
# --------------------------------------------------------------------------- #
def test_provider_chain_for(tmp_path):
    from reasoning_trajectories.config import Config
    from reasoning_trajectories.datasets.entailmentbank import EntailmentBank
    from reasoning_trajectories.reasoning import ReasoningProvider

    d = tmp_path / "dataset" / "task_1"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    ex = next(iter(EntailmentBank(task=1, split="validation", local_dir=tmp_path, shuffle_seed=None)))

    cfg = Config.load("configs/default.yaml", overrides={"reasoning.source": "dataset_field"})
    prov = ReasoningProvider(cfg.reasoning, model=None)
    chain = prov.chain_for(ex)
    assert chain.is_structured
    assert chain.text == ex.meta["reasoning"] == prov.reasoning_for(ex)
