#!/usr/bin/env python
"""Run (or reuse) an extraction and emit a small trajectory-geometry report.

    python studies/hidden_state_trajectories/run_study.py --config configs/default.yaml

Outputs go to  data/runs/<run_name>/report/ :
    summary.md                 accuracy, feature means by outcome, effect sizes
    features_token.png         feature histograms (correct vs incorrect)
    features_layer.png
    chain_<id>_profile.png     curvature/torsion along an example chain
    chain_<id>_pca.png         PCA view of the same chain (visualisation only)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reasoning_trajectories import Config, run_pipeline  # noqa: E402

import analysis  # noqa: E402  (same dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", help="skip extraction, analyse this existing run dir")
    ap.add_argument("--examples", type=int, default=3, help="how many chains to profile individually")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        cfg = Config.load(args.config)
        manifest = run_pipeline(cfg)
        run_dir = cfg.run_dir
        print(f"extraction done: acc={manifest['accuracy']}  ->  {run_dir}")

    cfg, store, features = analysis.load_run(run_dir)
    report = run_dir / "report"
    report.mkdir(exist_ok=True)

    lines = [f"# Trajectory study — {run_dir.name}", ""]
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        import json
        m = json.loads(manifest_path.read_text())
        lines += [
            f"- model: `{m['model']}` (remote={m['remote']}), L={m['n_layers']}, h={m['hidden_size']}",
            f"- anchor layer l*: **{m['star_layer']}**  (extra: {m['extra_token_layers']})",
            f"- examples: {m['n_examples']}, accuracy: {m['accuracy']}",
            "",
        ]

    for traj in ("token", "layer"):
        if not (features["trajectory"] == traj).any():
            continue
        lines += [f"## {traj} trajectory — mean feature by outcome", ""]
        lines += [describe_to_md(analysis.describe_by_outcome(features, traj)), ""]
        eff = analysis.outcome_effect_sizes(features, traj)
        if not eff.empty:
            lines += [f"### {traj} — effect size (correct − incorrect)", "",
                      describe_to_md(eff), ""]
        analysis.plot_feature_distributions(features, report / f"features_{traj}.png", traj)

    ids = [c["id"] for c in list(store.iter_chains())[: args.examples]]
    for ex_id in ids:
        try:
            analysis.plot_chain_profile(store, ex_id, cfg, report / f"chain_{safe(ex_id)}_profile.png")
            analysis.plot_trajectory_pca(store, ex_id, report / f"chain_{safe(ex_id)}_pca.png")
        except (KeyError, FileNotFoundError):
            continue
    lines += ["## Per-chain profiles", "", f"Profiled {len(ids)} chains: {ids}", ""]

    (report / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {report}")


def describe_to_md(df) -> str:
    try:
        return df.round(4).to_markdown()
    except Exception:
        return "```\n" + df.round(4).to_string() + "\n```"


def safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


if __name__ == "__main__":
    main()
