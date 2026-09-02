#!/usr/bin/env python
"""Run the full extraction pipeline from a config file.

    python scripts/run_extraction.py --config configs/default.yaml
    python scripts/run_extraction.py --config configs/default.yaml \
        --set model.remote=true --set dataset.limit=8 --set run_name=obqa_remote

Dotted --set overrides are parsed as YAML scalars (true/false/ints/floats/str).
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_trajectories import Config, run_pipeline  # noqa: E402


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", dest="overrides", metavar="key=value")
    args = ap.parse_args()

    cfg = Config.load(args.config, overrides=parse_overrides(args.overrides))
    manifest = run_pipeline(cfg)
    print("\n== run complete ==")
    for k in ("run_name", "model", "n_examples", "accuracy", "star_layer", "seconds"):
        print(f"  {k}: {manifest[k]}")
    print(f"  artifacts: {cfg.run_dir}")


if __name__ == "__main__":
    main()
