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
For output with the revision step, ``--pass second`` (default) describes the revised proof and a row the
reviser could not or did not revise has status ``revision:<why>``; ``--pass first`` describes the first
attempt. ``summary`` adds a first-vs-second comparison when the run used ``--judge both``; ``show --compare``
prints the first attempt next to the revised one.
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
def _lowest(scores: dict) -> int | None:
    present = [v for v in scores.values() if v is not None]
    return min(present) if len(present) == len(JUDGE_SCORES) else None


def describe(row: dict, i: int, min_score: int, use_pass: str = "second") -> dict:
    """Flat, plain-python view of one Data Designer row.

    Rows from the revision pipeline hold two attempts: the *first* (``tree`` / ``record_first`` /
    ``judge_first``) and the *second*, revised one (``tree_final`` / ``record`` / ``judge``). ``use_pass``
    picks which one the main columns describe. Older rows have a single attempt and ignore it."""
    new = "revision_status" in row
    which = use_pass if new else "first"
    rev_status = row.get("revision_status") if new else None
    if new and which == "first":
        rec, text = row.get("record_first"), row.get("proof_text_first")
        valid, errs, jcol, tree_col = bool(row.get("valid_first")), row.get("errors_first"), "judge_first", "tree"
    else:
        rec, text = row.get("record"), row.get("proof_text")
        valid, errs, jcol = bool(row.get("structure_valid")), row.get("structure_errors"), "judge"
        tree_col = "tree_final" if new else "tree"
    rec = rec if isinstance(rec, dict) and rec else None
    if rec is None or not text:                            # older/odd outputs: rebuild with the generator's own code
        try:
            rec, text, _ = G.build_record(row.get("record_uuid"), row.get("domain", ""), row.get("skeleton"), row.get(tree_col))
        except Exception:  # noqa: BLE001
            text = text or ""
    scores = judge_scores(row, jcol)
    lowest = _lowest(scores)
    if which == "second" and new and rev_status != "revised":
        status = f"revision:{rev_status}"
    elif not valid:
        status = "structure"
    elif lowest is None:
        status = "no-judge"
    elif lowest < min_score:
        status = "judge"
    else:
        status = "pass"
    binding = [k for k, v in scores.items() if v is not None and v == lowest] if lowest is not None else []

    def ok(valid_flag: bool, sc: dict) -> bool | None:      # did this attempt pass? None = not judged
        low = _lowest(sc)
        return False if not valid_flag else (None if low is None else low >= min_score)

    out = {
        "row": i, "id": (rec or {}).get("id") or row.get("record_uuid"), "record_uuid": row.get("record_uuid"),
        "domain": row.get("domain"), "topic": row.get("topic"), "question_style": row.get("question_style"),
        "n_steps": int(row["n_steps"]) if row.get("n_steps") not in (None, "") else None,
        "pass_shown": which, "status": status, "structure_valid": valid, "structure_errors": errs if errs else "",
        **{k: scores[k] for k in JUDGE_SCORES}, "lowest": lowest, "binding": ",".join(binding),
        "question": (rec or {}).get("question", ""), "answer": (rec or {}).get("answer", ""),
        "hypothesis": (rec or {}).get("hypothesis", ""), "proof_text": text or "",
        "skeleton_text": row.get("skeleton_text") or "", "_judge": G._as_obj(row.get(jcol)),
        "revision_status": rev_status, "revision_notes": row.get("revision_notes") or "" if new else "",
    }
    if new:                                                 # both attempts, for first-vs-second comparison
        first, second = judge_scores(row, "judge_first"), judge_scores(row, "judge")
        out["proof_text_first"] = row.get("proof_text_first") or ""
        out["_first_scores"], out["_second_scores"] = first, second
        out["first_ok"] = ok(bool(row.get("valid_first")), first)
        out["second_ok"] = False if rev_status != "revised" else ok(bool(row.get("structure_valid")), second)
    return out


def load(path: str, min_score: int, use_pass: str = "second") -> list[dict]:
    rows = load_rows(Path(path))
    if not rows:
        sys.exit(f"no rows found in {path}")
    return [describe(r, i, min_score, use_pass) for i, r in enumerate(rows)]


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def compare_passes(df, rows: list[dict], min_score: int) -> None:
    """First attempt vs revised proof, over the rows that were judged on both (``--judge both``)."""
    both = [r for r in rows if r.get("first_ok") is not None and r.get("second_ok") is not None
            and any(v is not None for v in r["_first_scores"].values())]
    if not both:
        print("\n(first vs second comparison needs a run with --judge both)")
        return
    n = len(both)
    f_ok = sum(r["first_ok"] for r in both)
    s_ok = sum(r["second_ok"] for r in both)
    fixed = sum((not r["first_ok"]) and r["second_ok"] for r in both)
    broken = sum(r["first_ok"] and not r["second_ok"] for r in both)
    print(f"\nfirst attempt vs revised proof ({n} rows judged on the first attempt; a row that could not be "
          f"revised counts as failing the second):")
    print(f"  pass rate        first {pct(f_ok, n)}   second {pct(s_ok, n)}")
    print(f"  fixed by revision (fail -> pass): {fixed}   broken by revision (pass -> fail): {broken}   "
          f"pass both: {sum(r['first_ok'] and r['second_ok'] for r in both)}   fail both: "
          f"{sum((not r['first_ok']) and (not r['second_ok']) for r in both)}")
    for label, key in (("first ", "_first_scores"), ("second", "_second_scores")):
        vals = {k: [r[key][k] for r in both if r[key][k] is not None] for k in JUDGE_SCORES}
        print(f"  mean {label}      " + "  ".join(
            f"{k}={sum(v) / len(v):.2f}" if v else f"{k}=n/a" for k, v in vals.items()))


def cmd_summary(a) -> int:
    import pandas as pd

    rows = load(a.path, a.min_judge_score, a.use_pass)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    n = len(df)
    st = Counter(df["status"])
    has_revision = df["revision_status"].notna().any()
    print(f"{n} rows in {a.path}" + (f" | showing the {a.use_pass} pass" if has_revision else ""))
    if a.requested:
        print(f"requested {a.requested}, saved {n}: {a.requested - n} missing "
              f"(dropped after retries, or the run was stopped; they are not in the file)")
    print(f"\nstatus (pass = structure ok and every score >= {a.min_judge_score}):")
    for s in sorted(st, key=lambda s: (STATUSES.index(s) if s in STATUSES else len(STATUSES), s)):
        print(f"  {s:<28}{pct(st[s], n)}")

    if has_revision:
        print("\nrevision status (what the reviser did with each first attempt):")
        for s, c in Counter(df["revision_status"].dropna()).most_common():
            print(f"  {s:<28}{pct(c, n)}")
        compare_passes(df, rows, a.min_judge_score)

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
    rows = load(a.path, a.min_judge_score, a.use_pass)
    picked = selected(rows, a)
    print(f"showing {len(picked)} of {len(rows)} rows")
    cut = (lambda s: s) if a.full else (lambda s: s if len(str(s)) <= 400 else str(s)[:400] + " ...")
    for r in picked:
        print("\n" + "=" * 100)
        shown = f" | {r['pass_shown']} pass" if r["revision_status"] else ""
        print(f"row {r['row']} | {r['id']} | {r['domain']} / {r['topic']} | {r['n_steps']} steps{shown} | status: {r['status'].upper()}")
        print(f"style: {r['question_style']}")
        if r["structure_errors"]:
            print(f"structure errors: {r['structure_errors']}")
        if r["revision_status"]:
            print(f"revision: {r['revision_status']}" + (f" | notes: {cut(r['revision_notes'])}" if r["revision_notes"] else ""))
            if a.compare and r.get("proof_text_first"):
                print("\n--- FIRST ATTEMPT (before revision) ---\n" + r["proof_text_first"]
                      + "\nfirst-attempt scores: " + ", ".join(f"{k}={v}" for k, v in r["_first_scores"].items()))
                print("--- shown below: " + r["pass_shown"] + " pass ---")
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

    rows = load(a.path, a.min_judge_score, a.use_pass)
    recs = []
    for r in rows:
        d = {k: v for k, v in r.items() if not k.startswith("_")}
        for k in JUDGE_SCORES:                              # revision pipeline: also give both attempts' scores
            if "_first_scores" in r:
                d[f"first_{k}"], d[f"second_{k}"] = r["_first_scores"][k], r["_second_scores"][k]
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
        p.add_argument("--pass", dest="use_pass", choices=("second", "first"), default="second",
                       help="for output with a revision step: describe the revised proof (second, default) or the "
                            "first attempt (first). Older output has one attempt and ignores this")

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
    w.add_argument("--compare", action="store_true", help="also print the first attempt (before revision) and its scores")
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
