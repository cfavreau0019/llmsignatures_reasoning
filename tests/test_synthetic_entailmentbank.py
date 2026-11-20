"""Pure-logic tests for scripts/synthetic_entailmentbank.py (no Data Designer, no API calls)."""
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthetic_entailmentbank as S  # noqa: E402

# a real EntailmentBank record (MSA_2015_5_8): 4 steps, depth 3, branched
REAL_PROOF = ("sent2 & sent5 -> int1: an earthquake usually occurs in a short amount of time; "
              "sent1 & sent3 -> int2: an earthquake can change earth's surface by shaking the ground; "
              "int1 & int2 -> int3: an earthquake can change earth's surface in a short amount of time; "
              "int3 & sent4 -> hypothesis; ")
REAL_LISP = "((((((sent2 sent5) -> int1) ((sent1 sent3) -> int2)) -> int3) sent4) -> int4)"
REAL_POLISH = "# int4 & # int3 & # int1 & sent2 sent5 # int2 & sent1 sent3 sent4"
LEAVES = {
    "sent1": "shaking the ground of the earth can change earth's surface",
    "sent2": "10 to 30 seconds is a short amount of time",
    "sent3": "an earthquake causes the ground to shake",
    "sent4": "in a short amount of time is similar to quickly",
    "sent5": "an earthquake usually occurs over a period of 10 to 30 seconds",
}


def test_encoders_reproduce_a_real_record():
    steps, ints = S.parse_steps(REAL_PROOF)
    assert len(steps) == 4 and steps[-1]["conclusion"] == "hypothesis"
    assert S.proof_string(steps, ints) == REAL_PROOF
    assert S.lisp_proof(steps) == REAL_LISP
    assert S.polish_proof(steps) == REAL_POLISH
    ft = S.full_text_proof(steps, LEAVES, {**ints, "int4": "an earthquake changes earth 's surface quickly"})
    assert ft.startswith(" [BECAUSE] 10 to 30 seconds is a short amount of time [AND] ")
    assert "[BECAUSE] int1 [AND] int2 [INFER] int3: " in ft and ft.endswith("int4: an earthquake changes earth 's surface quickly")
    g = S.graph_stats(steps)
    assert (g["depth"], g["n_steps"], g["shape"], g["n_leaves"]) == (3, 4, "branched", 5)


def test_clean_sentence_matches_corpus_convention():
    assert S.clean_sentence("  A Plant Needs Sunlight; and water: always. ") == "a plant needs sunlight, and water always"


@pytest.mark.parametrize("n_steps", range(1, 13))
def test_skeletons_assemble_into_self_consistent_records(n_steps):
    for seed in range(60):
        sk = S.build_skeleton(n_steps, random.Random(seed))
        assert sk["n_steps"] == n_steps and sk["steps"][-1]["conclusion"] == "hypothesis"
        leaves = {k: f"leaf {k} of case {seed}" for k in sk["leaf_ids"]}
        ints = {k: f"derived {k} of case {seed}" for k in sk["int_ids"]}
        rec = S.assemble_task1(rid="SYN_T_1", question="Q?", answer="A", hypothesis=f"goal of case {seed}",
                               steps=sk["steps"], leaf_text=leaves, int_text=ints)
        assert S.check_task1_record(rec) == []
        if n_steps >= 3:
            assert sk["shape"] in ("linear", "branched")
        assert (sk["shape"] == "single-step") == (n_steps == 1)


def test_skeleton_distribution_is_skewed_to_deeper_proofs_and_has_all_shapes():
    rng = random.Random(0)
    skels = [S.build_skeleton(S._weighted(rng, list(S.STEP_WEIGHTS.items())), random.Random(i)) for i in range(3000)]
    assert sum(s["n_steps"] >= 3 for s in skels) / len(skels) > 0.8          # real data: 50 %
    assert {"single-step", "linear", "branched"} == {s["shape"] for s in skels}
    assert 0.03 < sum(bool(s["reused_nodes"]) for s in skels) / len(skels) < 0.15   # real: 8 %
    assert all(f == 2 or f in (3, 4) for s in skels for f in s["fan_in"])


def test_build_record_flags_bad_llm_output():
    sk = S.build_skeleton(4, random.Random(1))
    leaves = [{"id": k, "text": f"Fact {k}."} for k in sk["leaf_ids"]]
    inter = [{"id": k, "text": f"Conclusion {k}"} for k in sk["int_ids"]]
    good = {"question": "Why?", "answer": "Because", "hypothesis": "The Goal.", "leaves": leaves, "intermediates": inter}
    rec, text, errs = S.build_record("abcd1234", "life science", sk, good)
    assert errs == [] and rec["id"] == "SYN_LS_abcd1234" and rec["hypothesis"] == "the goal" and "step 1:" in text
    for broken in (
        {**good, "leaves": leaves[:-1]},                       # a leaf id is missing
        {**good, "intermediates": []},                         # intermediates missing
        {**good, "hypothesis": ""},                            # empty hypothesis
        {**good, "leaves": leaves[:-1] + [{"id": leaves[-1]["id"], "text": leaves[0]["text"]}]},   # duplicate sentence
    ):
        _, _, errs = S.build_record("abcd1234", "life science", sk, broken)
        assert errs


# --------------------------------------------------------------------------- #
# revision pass: draft rendering and finalize()
# --------------------------------------------------------------------------- #
def _tree(sk, tag="a", **over):
    t = {"question": "Why?", "answer": "Because",
         "hypothesis": f"the goal {tag}",
         "leaves": [{"id": k, "text": f"Fact {k} {tag}."} for k in sk["leaf_ids"]],
         "intermediates": [{"id": k, "text": f"Conclusion {k} {tag}"} for k in sk["int_ids"]]}
    t.update(over)
    return t


def test_draft_text_is_tolerant_of_missing_pieces():
    sk = S.build_skeleton(4, random.Random(1))
    full = S.draft_text(sk, _tree(sk))
    assert "(MISSING)" not in full and full.count("step ") == 4 and "[hypothesis] the goal a" in full
    broken = _tree(sk, hypothesis="", leaves=_tree(sk)["leaves"][:-1])     # one leaf and the hypothesis missing
    text = S.draft_text(sk, broken)
    assert text.count("(MISSING)") >= 2 and "[hypothesis] (MISSING)" in text
    assert S.draft_text(sk, {}).count("(MISSING)") >= 4                   # nothing at all still renders


def test_finalize_uses_a_coherent_revision():
    sk = S.build_skeleton(4, random.Random(2))
    rev = {**_tree(sk, "b"), "coherent": True, "revision_notes": "dropped a redundant premise"}
    f = S.finalize("abcd1234", "life science", sk, _tree(sk, "a"), rev)
    assert f["status"] == "revised" and not f["errs"] and not f["first_errs"]
    assert f["rec"]["hypothesis"] == "the goal b" and f["first_rec"]["hypothesis"] == "the goal a"
    assert f["notes"] == "dropped a redundant premise" and f["final_tree"]["hypothesis"] == "the goal b"
    assert f["rec"]["meta"]["triples"] != f["first_rec"]["meta"]["triples"]      # the sentences really changed
    assert f["rec"]["proof"].replace("b", "a") == f["first_rec"]["proof"]        # ... but the graph did not


def test_finalize_reports_incoherent_and_broken_revisions_instead_of_hiding_them():
    sk = S.build_skeleton(4, random.Random(3))
    first = _tree(sk, "a")
    inc = S.finalize("abcd1234", "life science", sk, first, {**_tree(sk, "b"), "coherent": False, "revision_notes": "no valid chain"})
    assert inc["status"] == "incoherent" and inc["rec"] == inc["first_rec"]       # unchanged first attempt, flagged
    bad = S.finalize("abcd1234", "life science", sk, first,
                     {**_tree(sk, "b"), "coherent": True, "leaves": _tree(sk)["leaves"][:-1]})   # reviser lost an id
    assert bad["status"] == "invalid_revision" and bad["errs"] and not bad["first_errs"]
    assert S.finalize("abcd1234", "life science", sk, first, None)["status"] == "no_revision"
    assert S.finalize("abcd1234", "life science", sk, first, {})["status"] == "no_revision"


def test_finalize_can_rescue_a_broken_first_attempt():
    sk = S.build_skeleton(3, random.Random(4))
    broken = _tree(sk, "a", leaves=_tree(sk)["leaves"][:-1])
    f = S.finalize("abcd1234", "life science", sk, broken, {**_tree(sk, "b"), "coherent": True})
    assert f["first_errs"] and f["status"] == "revised" and not f["errs"]
    assert "(MISSING)" in f["first_text"]                                         # the reviser is shown what is absent
