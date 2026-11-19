"""Run the synthetic EntailmentBank Data Designer config through the Python API.

Why this exists: the ``data-designer preview`` / ``create`` CLI commands crash on import with
Typer >= 0.2x ("Option() got an unexpected keyword argument 'click_type'"). ``validate`` works, and
the Python API they wrap works, so this driver calls the API directly (same config, same engine).

    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py validate            # offline
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py preview -n 10       # ~20 API calls
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py create  -n 1400 \
        --dataset-name entailmentbank_synth_v1                                       # paid, asks first
    .venv/Scripts/python scripts/run_synthetic_entailmentbank.py check-models        # tiny calls

``preview`` keeps everything in memory, so stopping it loses the rows. ``create`` checkpoints every
``--buffer-size`` records (default 100) to ``artifacts/<dataset-name>/``; stopping it loses only the
unfinished batch, and ``create ... --resume always`` (same ``--buffer-size``) continues from the last
saved one.

Then export to EntailmentBank task files:

    .venv/Scripts/python scripts/export_synthetic_entailmentbank.py \
        --input <dataset path printed by `create`> --out data/synthetic/entailmentbank_synth_v1

``preview`` and ``create`` make billed calls with the OPENAI_API_KEY from your ``.env``. Two 10-record
previews cost about 4,400 and 4,900 tokens per record across the two models. The share of records that
clear the default judge filter (>= 4 on every score; step_validity is the binding one) was 70 % in one
preview and 20 % in the other, so plan on roughly 2x what you want to keep and re-measure on a larger
preview before a big ``create``.

``--max-restarts N`` (before the subcommand) sets how many times a call whose output cannot be parsed
is re-run before the row is dropped; it is unrelated to judge scores.

An empty OpenAI balance (HTTP 429 "no credits remaining") would otherwise be retried forever as a
rate limit; the driver reclassifies it as a quota error so the run fails fast with a clear message.

OpenAI sometimes answers ``500 server_error`` and succeeds on the identical retry; Data Designer only
retries 502/503/504, so the driver widens that to include 500 (``--no-retry-500`` turns it off).
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

TOKENS_PER_RECORD = 4_700            # mean of two 10-record previews (both models, incl. reasoning tokens)
CALLS_PER_RECORD = 2                 # one tree + one judge call (invalid rows skip the judge)


def retry_server_errors(max_retries: int = 5) -> bool:
    """Make Data Designer's HTTP layer retry HTTP 500 as well as 502/503/504.

    OpenAI intermittently answers a request with ``500 server_error`` and succeeds on the identical retry
    (seen with gpt-6-luna). Data Designer only retries 502/503/504 and offers no public setting for it,
    so one 500 fails a health check or drops a generated row. Its model factory looks ``RetryConfig`` up
    on every call, so substituting a subclass with a wider policy is enough. Uses a private module, so
    it is guarded: if Data Designer's internals change it warns and carries on with the defaults.
    """
    try:
        from dataclasses import dataclass

        from data_designer.engine.models.clients import retry as dd_retry

        base = dd_retry.RetryConfig
        if 500 in base().retryable_status_codes:
            return True
        retries = max_retries                              # a class body cannot read an enclosing name it also assigns

        @dataclass(frozen=True)
        class RetryConfigWith500(base):                    # type: ignore[misc, valid-type]
            max_retries: int = retries
            retryable_status_codes: frozenset = frozenset({500, 502, 503, 504})

        dd_retry.RetryConfig = RetryConfigWith500
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not enable HTTP 500 retries ({type(exc).__name__}: {exc}); using Data Designer defaults",
              file=sys.stderr)
        return False


_NO_CREDIT_PHRASES = ("no credits remaining", "credit_balance_exhausted", "add credits", "billing hard limit")


def fail_fast_on_no_credits() -> bool:
    """Make an out-of-credits 429 abort the run instead of being retried forever.

    OpenAI answers an empty balance with HTTP 429 ``"You have no credits remaining..."``. Data Designer
    recognises quota errors by phrase ("insufficient credits", "insufficient_quota", ...) but looks only at
    the error *message*, which for this wording matches none of them, so the 429 is classed as an ordinary
    rate limit and retried without end (an earlier preview spun for 30+ minutes on ~1,100 of them).
    Adding the phrases makes it a ``ModelQuotaExceededError``: the rows fail and the run stops with a
    clear message. Private module, so guarded like ``retry_server_errors``.
    """
    try:
        from data_designer.engine.models.clients import errors as dd_errors

        original = dd_errors._looks_like_quota_exceeded_error
        if getattr(original, "_eb_patched", False):
            return True

        def patched(text: str) -> bool:
            return original(text) or any(t in text for t in _NO_CREDIT_PHRASES)

        patched._eb_patched = True                         # type: ignore[attr-defined]
        dd_errors._looks_like_quota_exceeded_error = patched
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not enable fail-fast on empty credits ({type(exc).__name__}: {exc}); "
              f"an empty balance will be retried indefinitely", file=sys.stderr)
        return False


def new_designer(a, DataDesigner, **kwargs):
    """Build a DataDesigner and apply ``--max-restarts`` / ``--buffer-size`` (if given) via its RunConfig."""
    dd = DataDesigner(**kwargs)
    settings = {}
    if a.max_restarts is not None:
        settings["max_conversation_restarts"] = a.max_restarts
    if getattr(a, "buffer_size", None) is not None:
        settings["buffer_size"] = a.buffer_size
    if settings:
        from data_designer.config.run_config import RunConfig

        dd.set_run_config(RunConfig(**settings))
    return dd


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
    new_designer(a, DataDesigner).validate(builder)
    print("configuration is valid (offline check; no API calls made)")
    return 0


def cmd_check_models(a, G, DataDesigner) -> int:
    new_designer(a, DataDesigner).check_models(G.load_config_builder(), max_attempts=3, retry_backoff_seconds=5)
    print("all models responded")
    return 0


def cmd_preview(a, G, DataDesigner) -> int:
    res = new_designer(a, DataDesigner).preview(G.load_config_builder(), num_records=a.num_records)
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
    n_groups = -(-a.num_records // a.buffer_size)
    print(f"about to generate {a.num_records} records with '{gen}' (trees) and '{judge}' (judge)\n"
          f"  ~{a.num_records * CALLS_PER_RECORD:,} API calls, ~{est / 1e6:.1f}M tokens (estimate from a 10-record preview)\n"
          f"  checkpoint every {a.buffer_size} records ({n_groups} batch{'es' if n_groups != 1 else ''}); an "
          f"interrupted run loses at most the unfinished batch, `--resume always` continues from the last saved one\n"
          f"  artifacts -> {Path(a.artifact_path).resolve()} / {a.dataset_name}")
    if not a.yes:
        try:
            ok = input("this makes billed API calls. proceed? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:                                   # non-interactive and no --yes: never spend by default
            ok = False
        if not ok:
            print("aborted (no API calls were made)")
            return 1
    res = new_designer(a, DataDesigner, artifact_path=a.artifact_path).create(
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
    c.add_argument("--buffer-size", type=int, default=100, metavar="N",
                   help="records per checkpointed batch (Data Designer's own default is 1000, which saves nothing "
                        "until 1000 records finish). Must be the same when you --resume. Default 100")
    c.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    c.set_defaults(fn=cmd_create)

    ap.add_argument("--max-restarts", type=int, default=None, metavar="N",
                    help="how many times Data Designer may re-run an LLM call whose output fails to parse into "
                         "the expected structure before the row is dropped (its default is 5; 0 = never retry). "
                         "Each restart is a full extra billed call; it does not react to low judge scores")
    ap.add_argument("--no-retry-500", action="store_true",
                    help="do not add HTTP 500 to the status codes Data Designer retries")
    a = ap.parse_args()
    G, DataDesigner = load()
    if a.max_restarts is not None and a.max_restarts < 0:
        ap.error("--max-restarts must be >= 0")
    if getattr(a, "buffer_size", 1) < 1:
        ap.error("--buffer-size must be >= 1")
    fail_fast_on_no_credits()
    if not a.no_retry_500:
        retry_server_errors()
    return a.fn(a, G, DataDesigner)


if __name__ == "__main__":
    raise SystemExit(main())
