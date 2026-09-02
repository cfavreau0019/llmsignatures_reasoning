#!/usr/bin/env python
"""Recompute features.parquet from stored hidden states -- no model needed.

Use this to iterate on the geometry / guards without re-running extraction.

    python scripts/recompute_features.py --run data/runs/obqa_llama31_8b_smoke \
        --set geometry.plane_sin2_tol=1e-8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_trajectories.config import Config  # noqa: E402
from reasoning_trajectories.storage import RunStore  # noqa: E402
from reasoning_trajectories.trajectories import Guards, summarize_trajectory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--set", action="append", dest="overrides", metavar="key=value")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    for p in args.overrides or []:
        k, v = p.split("=", 1)
        parts = k.strip().split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], yaml.safe_load(v))

    guards = Guards.from_config(cfg.geometry)
    store = RunStore(run_dir, store_dtype=cfg.extraction.store_dtype)
    chains = {c["id"]: c for c in store.iter_chains()}

    rows = []
    for example_id, chain in chains.items():
        arr = store.load_arrays(example_id)
        base = {
            "id": example_id,
            "answer_key": chain.get("answer_key"),
            "predicted": chain.get("predicted"),
            "is_correct": chain.get("is_correct"),
            "n_chain_tokens": len(chain.get("tokens") or []),
        }
        star = int(chain["star_layer"])
        for key in arr:
            if key.startswith("token_layer_"):
                l = int(key.rsplit("_", 1)[1])
                tag = "token" if l == star else f"token_L{l}"
                rows.append({**base, "trajectory": tag, "layer": l,
                             **summarize_trajectory(arr[key].astype(np.float64), guards)})
        if "layer_trajectory" in arr:
            rows.append({**base, "trajectory": "layer", "layer": -1,
                         **summarize_trajectory(arr["layer_trajectory"].astype(np.float64), guards)})

    out = store.write_features(rows)
    print(f"wrote {len(rows)} feature rows -> {out}")


if __name__ == "__main__":
    main()
