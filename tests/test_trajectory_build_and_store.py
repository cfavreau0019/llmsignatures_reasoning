"""Plumbing test that does not need torch / nnsight: fake per-layer hidden
states -> trajectories -> RunStore round trip -> feature recompute."""
import numpy as np

from reasoning_trajectories.extraction.record import ExtractionRecord
from reasoning_trajectories.storage import RunStore
from reasoning_trajectories.trajectories import (
    build_layer_trajectory,
    build_token_trajectory,
    summarize_trajectory,
)


def _fake_hidden(n_layers=6, n_pos=12, h=16, seed=0):
    rng = np.random.default_rng(seed)
    return {l: rng.standard_normal((n_pos, h)) for l in range(n_layers + 1)}


def test_build_token_and_layer_trajectories():
    hbl = _fake_hidden()
    star = 4
    tok = build_token_trajectory(hbl, star_layer=star, span=(3, None))
    assert tok.points.shape == (12 - 3, 16)
    assert tok.layer == star

    lay = build_layer_trajectory(hbl, span=(3, None))
    assert lay.points.shape == (7, 16)          # L+1 points
    assert list(lay.layer_indices) == list(range(7))

    f = summarize_trajectory(lay.points)
    assert 0.0 <= f["straightness"] <= 1.0


def test_runstore_roundtrip(tmp_path):
    hbl = _fake_hidden()
    tok = build_token_trajectory(hbl, star_layer=4, span=None)
    lay = build_layer_trajectory(hbl, span=None)
    rec = ExtractionRecord(
        example_id="ex-1", prompt="p", reasoning="r Answer: B", full_text="p r",
        tokens=["a"] * 12, n_positions=12, chain_start=0, star_layer=4,
        token_trajectories={4: tok}, layer_trajectory=lay,
        answer_key="B", predicted="B",
    )
    assert rec.is_correct is True

    store = RunStore(tmp_path / "run", store_dtype="float16")
    store.save_record(rec)
    arr = store.load_arrays("ex-1")
    assert arr["token_layer_4"].shape == (12, 16)
    assert arr["layer_trajectory"].shape == (7, 16)
    assert list(store.iter_chains())[0]["predicted"] == "B"

    rows = [{"id": "ex-1", "trajectory": "token", "layer": 4,
             **summarize_trajectory(arr["token_layer_4"].astype(np.float64))}]
    out = store.write_features(rows)
    assert out.exists()
