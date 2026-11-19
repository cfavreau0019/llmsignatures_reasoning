"""Look inside Data Designer output for the synthetic EntailmentBank config: how often rows failed, why,
and what the text says. Reads a preview ``.parquet`` / ``.jsonl`` or a ``create`` dataset directory.
No API calls, nothing is modified.

    .venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py summary artifacts/synthetic_entailmentbank_preview_10.parquet
    .venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py show    <path> --failed -n 3
    .venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py flat    <path> out.csv     # table for PyCharm / Excel

``summary``  pass/fail counts, why rows failed, score distribution, which score binds, pass rate by number
             of steps / domain / question style. ``--requested N`` also reports the shortfall (rows that
             Data Designer dropped after retries are not in the file at all).
``show``     records as readable text: question, hypothesis, proof steps, the judge's score and reasoning.
             Filters: ``--failed`` / ``--passed``, ``--id``, ``--domain``, ``--n-steps``, ``--below N``
             (with ``--score NAME`` to test one score instead of the lowest), ``--sample`` for random rows.
``flat``     one row per record with plain columns (no nested values): ids, status, the four scores and the
             judge's reasoning for each, question, answer, hypothesis, proof text. ``.csv`` or ``.parquet``.

``dump``     the raw rows, every column unchanged (nested values intact), as an indented ``.json`` array or
             ``.jsonl``, to read in an editor.

A row's **status** is ``structure`` (the rebuilt record failed a structural check), ``no-judge`` (no usable
scores), ``judge`` (lowest score below ``--min-judge-score``, default 4, the exporter's default), or ``pass``.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

for stream in (sys.stdout, sys.stderr):                    # proof text and Data Designer output are not cp1252-safe
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_entailmentbank as G  # noqa: E402
from export_synthetic_entailmentbank import JUDGE_SCORES, judge_scores, load_rows, to_py  # noqa: E402

STATUSES = ("pass", "judge", "no-judge", "structure")


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def describe(row: dict, i: int, min_score: int) -> dict:
    """Flat, plain-python view of one Data Designer row."""
    rec = row.get("record") if isinstance(row.get("record"), dict) else None
    text = row.get("proof_text")
    if rec is None or not text:                            # older/odd outputs: rebuild with the generator's own code
        try:
            rec, text, _ = G.build_record(row.get("record_uuid"), row.get("domain", ""), row.get("skeleton"), row.get("tree"))
        except Exception:  # noqa: BLE001
            rec, text = rec, text or ""
    scores = judge_scores(row)
    present = [v for v in scores.values() if v is not None]
    valid = bool(row.get("structure_valid"))
    lowest = min(present) if len(present) == len(JUDGE_SCORES) else None
    if not valid:
        status = "structure"
    elif lowest is None:
        status = "no-judge"
    elif lowest < min_score:
        status = "judge"
    else:
        status = "pass"
    binding = [k for k, v in scores.items() if v is not None and v == lowest] if lowest is not None else []
    errs = row.get("structure_errors")
    return {
        "row": i, "id": (rec or {}).get("id") or row.get("record_uuid"), "record_uuid": row.get("record_uuid"),
        "domain": row.get("domain"), "topic": row.get("topic"), "question_style": row.get("question_style"),
        "n_steps": int(row["n_steps"]) if row.get("n_steps") not in (None, "") else None,
        "status": status, "structure_valid": valid, "structure_errors": errs if errs else "",
        **{k: scores[k] for k in JUDGE_SCORES}, "lowest": lowest, "binding": ",".join(binding),
        "question": (rec or {}).get("question", ""), "answer": (rec or {}).get("answer", ""),
        "hypothesis": (rec or {}).get("hypothesis", ""), "proof_text": text or "",
        "skeleton_text": row.get("skeleton_text") or "", "_judge": G._as_obj(row.get("judge")),
    }


def load(path: str, min_score: int) -> list[dict]:
    rows = load_rows(Path(path))
    if not rows:
        sys.exit(f"no rows found in {path}")
    return [describe(r, i, min_score) for i, r in enumerate(rows)]


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def cmd_summary(a) -> int:
    import pandas as pd

    rows = load(a.path, a.min_judge_score)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_judge"} for r in rows])
    n = len(df)
    st = Counter(df["status"])
    print(f"{n} rows in {a.path}")
    if a.requested:
        print(f"requested {a.requested}, saved {n}: {a.requested - n} missing "
              f"(dropped after retries, or the run was stopped; they are not in the file)")
    print(f"\nstatus (pass = structure ok and every score >= {a.min_judge_score}):")
    for s in STATUSES:
        print(f"  {s:<10}{pct(st.get(s, 0), n)}")

    if st.get("structure"):
        print("\nstructure errors:")
        for msg, c in Counter(str(e)[:110] for e in df.loc[df.status == "structure", "structure_errors"]).most_common(8):
            print(f"  {c:>3}  {msg}")

    scored = df[df["lowest"].notna()]
    if len(scored):
        print(f"\njudge scores ({len(scored)} judged rows; count per score):")
        dist = pd.DataFrame({k: scored[k].value_counts() for k in JUDGE_SCORES}).reindex(range(1, 6)).fillna(0).astype(int)
        dist.index.name = "score"
        print(dist.to_string())
        print("  mean   " + "  ".join(f"{k}={scored[k].mean():.2f}" for k in JUDGE_SCORES))
        low = scored[scored["lowest"] < a.min_judge_score]
        if len(low):
            print(f"\nwhich score is below {a.min_judge_score} in the {len(low)} judge-failed rows "
                  f"(a row can count under several):")
            for k in JUDGE_SCORES:
                print(f"  {k:<22}{pct(int((low[k] < a.min_judge_score).sum()), len(low))}")

    for col in ("n_steps", "domain", "question_style"):
        sub = df[df[col].notna()]
        if sub[col].nunique() < 2:
            continue
        g = sub.assign(ok=sub["status"] == "pass").groupby(col)["ok"].agg(["sum", "count"])
        print(f"\npass rate by {col}:")
        width = max(len(str(k)) for k in g.index) + 2
        for key, r in g.iterrows():
            print(f"  {str(key):<{width}}{pct(int(r['sum']), int(r['count']))}")
    return 0


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
def selected(rows: list[dict], a) -> list[dict]:
    out = []
    for r in rows:
        if a.failed and r["status"] == "pass":
            continue
        if a.passed and r["status"] != "pass":
            continue
        if a.id and not any(x in str(r["id"]) or x == str(r["row"]) for x in a.id):
            continue
        if a.domain and a.domain.lower() not in str(r["domain"]).lower():
            continue
        if a.n_steps is not None and r["n_steps"] != a.n_steps:
            continue
        if a.below is not None:
            v = r.get(a.score) if a.score else r["lowest"]
            if v is None or v >= a.below:
                continue
        out.append(r)
    if a.sample:
        random.Random(a.seed).shuffle(out)
    return out[a.offset:a.offset + a.n]


def cmd_show(a) -> int:
    rows = load(a.path, a.min_judge_score)
    picked = selected(rows, a)
    print(f"showing {len(picked)} of {len(rows)} rows")
    cut = (lambda s: s) if a.full else (lambda s: s if len(str(s)) <= 400 else str(s)[:400] + " ...")
    for r in picked:
        print("\n" + "=" * 100)
        print(f"row {r['row']} | {r['id']} | {r['domain']} / {r['topic']} | {r['n_steps']} steps | status: {r['status'].upper()}")
        print(f"style: {r['question_style']}")
        if r["structure_errors"]:
            print(f"structure errors: {r['structure_errors']}")
        print(f"\nQ: {r['question']}\nA: {r['answer']}\nHYPOTHESIS: {r['hypothesis']}\n")
        print(r["proof_text"] if a.full else "\n".join(cut(l) for l in str(r["proof_text"]).splitlines()))
        print("\njudge:")
        for k in JUDGE_SCORES:
            j = r["_judge"].get(k) if isinstance(r["_judge"], dict) else None
            reason = j.get("reasoning") if isinstance(j, dict) else None
            print(f"  {k:<22}{r[k]}  {cut(reason) if reason else ''}")
    return 0


# --------------------------------------------------------------------------- #
# flat
# --------------------------------------------------------------------------- #
def cmd_flat(a) -> int:
    import pandas as pd

    rows = load(a.path, a.min_judge_score)
    recs = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "_judge"}
        for k in JUDGE_SCORES:
            j = r["_judge"].get(k) if isinstance(r["_judge"], dict) else None
            d[f"{k}_reasoning"] = j.get("reasoning") if isinstance(j, dict) else None
        recs.append(d)
    df = pd.DataFrame(recs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")  # BOM so Excel reads UTF-8
    print(f"wrote {len(df)} rows x {df.shape[1]} columns -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def cmd_dump(a) -> int:
    import json

    rows = [to_py(r) for r in load_rows(Path(a.path))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".jsonl":                     # one compact record per line
        out.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    else:                                                  # one indented JSON array: fold/unfold it in an editor
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows, every column, nested values intact -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("path", help="preview .parquet/.jsonl, or a `create` dataset directory")
        p.add_argument("--min-judge-score", type=int, default=4, help="pass threshold on every score (default 4)")

    s = sub.add_parser("summary", help="failure counts, reasons, score distribution, pass rates")
    common(s)
    s.add_argument("--requested", type=int, default=None, help="rows you asked for, to report the shortfall")
    s.set_defaults(fn=cmd_summary)

    w = sub.add_parser("show", help="print records as readable text")
    common(w)
    w.add_argument("-n", type=int, default=5, help="how many rows (default 5)")
    w.add_argument("--offset", type=int, default=0)
    w.add_argument("--failed", action="store_true", help="only rows that did not pass")
    w.add_argument("--passed", action="store_true", help="only rows that passed")
    w.add_argument("--id", nargs="+", help="id substring(s) or row number(s)")
    w.add_argument("--domain", help="substring of the domain, e.g. 'earth'")
    w.add_argument("--n-steps", type=int, default=None)
    w.add_argument("--below", type=int, default=None, help="only rows whose lowest score is < N")
    w.add_argument("--score", choices=JUDGE_SCORES, default=None, help="with --below: test this score, not the lowest")
    w.add_argument("--sample", action="store_true", help="random rows instead of the first ones")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--full", action="store_true", help="do not truncate long lines")
    w.set_defaults(fn=cmd_show)

    f = sub.add_parser("flat", help="write a plain table (csv or parquet) for PyCharm / Excel")
    common(f)
    f.add_argument("out", help="output path ending in .csv or .parquet")
    f.set_defaults(fn=cmd_flat)

    d = sub.add_parser("dump", help="write the raw rows, all columns unchanged, as .json (indented) or .jsonl")
    d.add_argument("path", help="preview .parquet/.jsonl, or a `create` dataset directory")
    d.add_argument("out", help="output path ending in .json (indented array) or .jsonl (one row per line)")
    d.set_defaults(fn=cmd_dump)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
