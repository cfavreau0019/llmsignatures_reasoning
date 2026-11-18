"""Run the synthetic EntailmentBank Data Designer config through the Python API.

Why this exists: the ``data-designer preview`` / ``create`` CLI commands crash on import with
Typer >= 0.2x ("Option() got an unexpected keyword argument 'click_type'"). ``validate`` works, and
the Python API they wrap works, so this driver calls the API directly (same config, same engine).

    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py validate            # offline
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py preview -n 10       # ~20 API calls
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py create  -n 1400 \
        --dataset-name entailmentbank_synth_v1                                       # paid, asks first
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py check-models        # tiny calls

Then export to EntailmentBank task files:

    .venv/Scripts/python scripts/export_synthetic_entailmentbank.py \
        --input <dataset path printed by `create`> --out data/synthetic/entailmentbank_synth_v1

``preview`` and ``create`` make billed calls with the OPENAI_API_KEY from your ``.env``. In the
10-record preview each record cost roughly 4,400 tokens across the two models, and about 30 % of
records fail the default judge filter (>= 4 on every score), so request ~1.4x what you want to keep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):                   # Data Designer prints emoji; Windows consoles default to cp1252
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

TOKENS_PER_RECORD = 4_400            # measured in the 10-record preview (both models, incl. reasoning tokens)
CALLS_PER_RECORD = 2                 # one tree + one judge call (invalid rows skip the judge)


def load():
    try:
        import synthetic_entailmentbank as G
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        sys.exit(f"{exc}\nRun this with the interpreter that has data-designer installed "
                 f"(e.g. .venv/Scripts/python); pip install data-designer")
    return G, DataDesigner


def summarize(df) -> None:
    for i, r in df.iterrows():
        rec = r.get("record") if isinstance(r.get("record"), dict) else {}
        j = r.get("judge")
        scores = ({k: (v.get("score") if isinstance(v, dict) else v) for k, v in j.items()}
                  if isinstance(j, dict) else None)
        print(f"[{i}] {rec.get('id')} | {r['domain']} | steps={r['n_steps']} | valid={r['structure_valid']} "
              f"| judge={scores}" + (f" | errs={str(r['structure_errors'])[:100]}" if not r["structure_valid"] else ""))


def cmd_validate(a, G, DataDesigner) -> int:
    builder = G.load_config_builder()
    DataDesigner().validate(builder)
    print("configuration is valid (offline check; no API calls made)")
    return 0


def cmd_check_models(a, G, DataDesigner) -> int:
    DataDesigner().check_models(G.load_config_builder())
    print("all models responded")
    return 0


def cmd_preview(a, G, DataDesigner) -> int:
    res = DataDesigner().preview(G.load_config_builder(), num_records=a.num_records)
    df = res.dataset
    out = Path(a.out) if a.out else ROOT / "artifacts" / f"synthetic_entailmentbank_preview_{a.num_records}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nsaved {len(df)} rows -> {out}")
    summarize(df)
    print(f"\nnext: python scripts/export_synthetic_entailmentbank.py --input {out} --out <dir>")
    return 0


def cmd_create(a, G, DataDesigner) -> int:
    from data_designer.config.run_config import ResumeMode

    builder = G.load_config_builder()
    gen = [mc for mc in builder.model_configs if mc.alias == G.GEN_ALIAS][0].model
    judge = [mc for mc in builder.model_configs if mc.alias == G.JUDGE_ALIAS][0].model
    est = a.num_records * TOKENS_PER_RECORD
    print(f"about to generate {a.num_records} records with '{gen}' (trees) and '{judge}' (judge)\n"
          f"  ~{a.num_records * CALLS_PER_RECORD:,} API calls, ~{est / 1e6:.1f}M tokens (estimate from a 10-record preview)\n"
          f"  artifacts -> {Path(a.artifact_path).resolve()} / {a.dataset_name}")
    if not a.yes:
        try:
            ok = input("this makes billed API calls. proceed? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:                                   # non-interactive and no --yes: never spend by default
            ok = False
        if not ok:
            print("aborted (no API calls were made)")
            return 1
    res = DataDesigner(artifact_path=a.artifact_path).create(
        builder, num_records=a.num_records, dataset_name=a.dataset_name, resume=ResumeMode(a.resume))
    print(f"\nwrote {res.actual_num_records} records{' (PARTIAL)' if res.is_partial else ''} -> {res.dataset_path}")
    print(f"next: python scripts/export_synthetic_entailmentbank.py --input {res.dataset_path} "
          f"--out data/synthetic/{a.dataset_name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check the config offline").set_defaults(fn=cmd_validate)
    sub.add_parser("check-models", help="send a tiny request to each model").set_defaults(fn=cmd_check_models)

    p = sub.add_parser("preview", help="generate a few records in memory and save them as parquet")
    p.add_argument("-n", "--num-records", type=int, default=10)
    p.add_argument("--out", default=None, help="parquet path (default artifacts/..._preview_N.parquet)")
    p.set_defaults(fn=cmd_preview)

    c = sub.add_parser("create", help="generate a full dataset to disk (billed)")
    c.add_argument("-n", "--num-records", type=int, required=True)
    c.add_argument("--dataset-name", default="entailmentbank_synth")
    c.add_argument("--artifact-path", default=str(ROOT / "artifacts"))
    c.add_argument("--resume", default="never", choices=["never", "always", "if_possible"],
                   help="continue an interrupted run of the same dataset name")
    c.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    c.set_defaults(fn=cmd_create)

    a = ap.parse_args()
    G, DataDesigner = load()
    return a.fn(a, G, DataDesigner)


if __name__ == "__main__":
    raise SystemExit(main())
