import json
import random
from pathlib import Path

import pytest

from reasoning_trajectories.config import Config
from reasoning_trajectories.datasets import PairedCorruption, load_dataset_from_config
from reasoning_trajectories.datasets.entailmentbank import EntailmentBank, resolve_steps
from reasoning_trajectories.datasets.perturbations import (
    CYCLE_DEFAULT,
    DonorPool,
    apply_corruption,
)

SAMPLE = Path(__file__).parent / "data" / "entailmentbank_task_1_dev.jsonl"
RECORDS = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _dataset(tmp_path, **kw):
    d = tmp_path / "dataset" / "task_1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    return EntailmentBank(task=1, split="validation", local_dir=tmp_path, shuffle_seed=None, **kw)


def _examples(tmp_path, **kw):
    return list(_dataset(tmp_path, **kw))


# --------------------------------------------------------------------------- #
# corruption primitives
# --------------------------------------------------------------------------- #
def _steps(rec):
    return resolve_steps(rec)


def test_step_shuffle_is_non_identity_and_preserves_step_set():
    steps = _steps(RECORDS[4])  # 4 steps
    rng = random.Random(0)
    new, info = apply_corruption(steps, "step_shuffle", rng=rng, donors=DonorPool([]), src_id="x")
    assert info["kind"] == "step_shuffle"
    assert info["permutation"] != sorted(info["permutation"])  # actually reordered
    assert [s["conclusion"] for s in new] != [s["conclusion"] for s in steps]
    assert sorted(s["conclusion"] for s in new) == sorted(s["conclusion"] for s in steps)


def test_short_chain_escalates_to_premise_substitution(tmp_path):
    exs = _examples(tmp_path)
    donors = DonorPool(exs)
    two_step = _steps(RECORDS[0])  # 2 steps
    rng = random.Random(0)
    _new, info = apply_corruption(two_step, "step_shuffle", rng=rng, donors=donors,
                                  src_id="EB_TEST_001", min_steps=3)
    assert info["kind"] == "premise_substitution"


def test_premise_substitution_other_tree_injects_foreign_sentence(tmp_path):
    exs = _examples(tmp_path)
    donors = DonorPool(exs)
    steps = _steps(RECORDS[2])
    own = {p for s in steps for p in s["premises"]} | {s["conclusion"] for s in steps}
    rng = random.Random(1)
    new, info = apply_corruption(steps, "premise_substitution_other_tree", rng=rng,
                                 donors=donors, src_id="EB_TEST_003")
    assert info["kind"] == "premise_substitution_other_tree"
    assert info["new"] not in own
    assert new[info["step"] - 1]["premises"][info["premise_index"]] == info["new"]


def test_conclusion_substitution_changes_one_conclusion_and_propagates(tmp_path):
    exs = _examples(tmp_path)
    donors = DonorPool(exs)
    steps = _steps(RECORDS[4])  # linear chain int1 -> int2 -> int3 -> hypothesis
    rng = random.Random(2)
    new, info = apply_corruption(steps, "conclusion_substitution", rng=rng,
                                 donors=donors, src_id="EB_TEST_005")
    si = info["step"] - 1
    changed_concl = [i for i, (a, b) in enumerate(zip(steps, new)) if a["conclusion"] != b["conclusion"]]
    assert changed_concl == [si]
    assert new[si]["conclusion_id"] == "SUB"
    # the substituted intermediate is reused by the next step -> premise rewritten
    assert info["propagated_to"] >= 1
    assert info["new"] in new[si + 1]["premises"]


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #
def test_paired_emits_matched_pairs_with_round_robin(tmp_path):
    exs = _examples(tmp_path)
    pc = PairedCorruption(exs, min_steps=3, seed=0)
    # only EB_TEST_003/004/005 have >= 3 steps
    assert len(pc) == 6
    out = list(pc)
    assert [e.meta["condition"] for e in out] == ["valid", "corrupted"] * 3

    valid = [e for e in out if e.meta["condition"] == "valid"]
    corrupt = [e for e in out if e.meta["condition"] == "corrupted"]
    assert {e.meta["pair_id"] for e in valid} == {e.meta["pair_id"] for e in corrupt}
    assert len({e.id for e in out}) == len(out)                 # ids unique
    # round-robin over the three kinds, one each
    assert sorted(e.meta["corruption"]["kind"] for e in corrupt) == sorted(CYCLE_DEFAULT)
    # corrupted reasoning text actually differs from its valid twin
    by_pair = {}
    for e in out:
        by_pair.setdefault(e.meta["pair_id"], {})[e.meta["condition"]] = e.meta["reasoning"]
    assert all(d["valid"] != d["corrupted"] for d in by_pair.values())


def test_paired_is_deterministic(tmp_path):
    a = list(PairedCorruption(_examples(tmp_path), seed=0))
    b = list(PairedCorruption(_examples(tmp_path), seed=0))
    assert [e.meta["reasoning"] for e in a] == [e.meta["reasoning"] for e in b]
    c = list(PairedCorruption(_examples(tmp_path), seed=1))
    assert [e.meta["reasoning"] for e in a] != [e.meta["reasoning"] for e in c]


def test_paired_emit_modes(tmp_path):
    exs = _examples(tmp_path)
    assert [e.meta["condition"] for e in PairedCorruption(exs, emit="valid")] == ["valid"] * 3
    assert [e.meta["condition"] for e in PairedCorruption(exs, emit="corrupted")] == ["corrupted"] * 3


def test_load_from_config_wraps_when_perturbation_set(tmp_path):
    d = tmp_path / "dataset" / "task_1"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = Config.load("configs/default.yaml", overrides={
        "dataset.name": "entailmentbank",
        "dataset.split": "validation",
        "dataset.shuffle_seed": 0,
        "dataset.params": {"task": 1, "local_dir": str(tmp_path),
                           "perturbation": {"min_steps": 3, "seed": 0}},
    })
    ds = load_dataset_from_config(cfg.dataset)
    assert isinstance(ds, PairedCorruption)
    conds = [e.meta["condition"] for e in ds]
    assert conds.count("valid") == 3 and conds.count("corrupted") == 3
