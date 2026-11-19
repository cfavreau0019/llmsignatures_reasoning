# Synthetic EntailmentBank — generate, run, export

How to generate EntailmentBank-style data with NVIDIA Data Designer and an OpenAI model, what each
script and option does, and what to do when a run misbehaves. The real data's format is documented in
[`entailmentbank_schema.md`](entailmentbank_schema.md); the synthetic files use the same format and
the same [JSON Schemas](schemas/).

| Script | Role |
|---|---|
| [`scripts/synthetic_entailmentbank.py`](../scripts/synthetic_entailmentbank.py) | The Data Designer configuration (`load_config_builder()`) plus pure helpers for the proof graph and the proof encodings. Not run directly. |
| [`scripts/run_synthetic_entailmentbank.py`](../scripts/run_synthetic_entailmentbank.py) | **Driver.** `validate`, `check-models`, `preview`, `create`. Used instead of the `data-designer` CLI (see [Known problems](#known-problems)). |
| [`scripts/export_synthetic_entailmentbank.py`](../scripts/export_synthetic_entailmentbank.py) | Turns Data Designer output into `task_1/2/3` files, filtering on structure and judge scores. |
| [`tests/test_synthetic_entailmentbank.py`](../tests/test_synthetic_entailmentbank.py) | Offline tests of the pure helpers (no API calls). |

## 1. Setup

```bash
.venv/Scripts/python -m pip install data-designer      # NVIDIA Data Designer (1.0.1 was used)
```

Put these in a `.env` file in the repo root (it is gitignored; the script loads it itself because the
Data Designer CLI does not):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-...          # writes the proofs
OPENAI_JUDGE_MODEL=gpt-...    # scores them (falls back to OPENAI_MODEL)
```

An `openai/` prefix on a model name is accepted and stripped. The account behind the key needs credits:
see [Out of credits](#out-of-credits-http-429).

## 2. How a record is made

**One row = one complete record** (one question with its whole proof tree), not one row per leaf or step.

1. Samplers fix the **domain**, **topic**, **question style** and the number of proof steps `n_steps`
   (1-12, skewed toward 3+ steps; real data is about 50 % at 3+).
2. A custom column builds the **proof skeleton** — the graph only: leaf ids, intermediate ids, which
   nodes feed which step. Shapes are single-step, linear, or branched (at least one step joining two
   derived nodes); fan-in is mostly 2 premises; about 8 % of skeletons reuse a leaf in two steps.
3. **One LLM call** writes every sentence for that graph: hypothesis first, then the leaves, then the
   intermediates.
4. **Deterministic code** assembles the raw task_1 record (`proof`, `lisp_proof`, `polish_proof`,
   `full_text_proof`, `meta`) and checks every structural invariant (`structure_valid`).
5. A **judge model** scores the record 1-5 on `step_validity`, `factual_correctness`,
   `answer_consistency` and `non_triviality`. Rows that fail the structure check skip the judge.

The proof graph is guaranteed well-formed; **whether each step is actually entailed is not** — it is only
judge-scored. `step_validity` is the score that rejects most rows.

Cost: about 2 API calls and roughly 4,400-4,900 tokens per record across both models (measured on two
10-record previews).

## 3. Running it

Run everything with the interpreter that has `data-designer` installed. Nothing here uses the CLI's
`preview` / `create`.

```bash
.venv/Scripts/python scripts/run_synthetic_entailmentbank.py validate
```
Offline check of the configuration. No API calls.

```bash
.venv/Scripts/python scripts/run_synthetic_entailmentbank.py check-models
```
Sends one tiny request to each model (3 attempts, 5 s backoff). Fails in seconds if the key, model name or balance is wrong.

```bash
.venv/Scripts/python scripts/run_synthetic_entailmentbank.py preview -n 10
```
Generates `n` records **in memory** and saves them to `artifacts/synthetic_entailmentbank_preview_<n>.parquet`
(or `--out PATH`), then prints a one-line summary per row. Billed. **If you stop a preview, its rows are lost** —
nothing is written until it finishes.

```bash
.venv/Scripts/python scripts/run_synthetic_entailmentbank.py create -n 1400 --dataset-name entailmentbank_synth_v1
```
Generates `n` records to disk under `artifacts/<dataset-name>/`. Prints the models, an estimated call and
token count, and asks `proceed? [y/N]` (no input, or EOF, means no). Billed.

Plan on roughly **2x** the records you want to keep: in two 10-record previews 70 % and 20 % of rows cleared the
default judge filter. Measure the rate on a ~50-record preview before a large `create`.

### Driver options

Global options go **before** the subcommand (`run_synthetic_entailmentbank.py --max-restarts 2 create ...`).

| Option | Applies to | Meaning |
|---|---|---|
| `--max-restarts N` | all subcommands | How many times Data Designer may re-run an LLM call whose output **fails to parse** into the expected structure before the row is dropped. Data Designer's default is 5; `0` disables. Each restart is a full extra billed call. It does **not** react to low judge scores. |
| `--no-retry-500` | all subcommands | By default the driver makes Data Designer also retry HTTP **500** (it retries only 502/503/504), up to 5 times. This turns that off. |
| `-n`, `--num-records` | `preview`, `create` | Number of rows. Required for `create`; `preview` defaults to 10. |
| `--out PATH` | `preview` | Parquet path for the saved preview. |
| `--dataset-name NAME` | `create` | Directory name under the artifact path (default `entailmentbank_synth`). Reuse the same name to resume. |
| `--artifact-path DIR` | `create` | Root for run output (default `artifacts/`, gitignored). |
| `--buffer-size N` | `create` | Records per **checkpointed batch**, default **100** (Data Designer's own default is 1000). |
| `--resume {never,always,if_possible}` | `create` | What to do with an interrupted run of the same dataset name (see below). Default `never`. |
| `-y`, `--yes` | `create` | Skip the billing confirmation. |

### Stopping and resuming

`create` saves a batch only when all `--buffer-size` records of it have finished. If a run stops (Ctrl-C, crash,
early shutdown), finished batches are kept and the unfinished one is discarded.

```bash
.venv/Scripts/python scripts/run_synthetic_entailmentbank.py create -n 1400 --dataset-name entailmentbank_synth_v1 --resume always
```

* `--resume always` continues from the last saved batch; `if_possible` does the same only if the configuration is unchanged
  and otherwise starts fresh. Use the **same `--buffer-size`** as the original run.
* `-n` may be equal to or larger than what is already saved (you can extend a dataset), not smaller.
* If the run ends with fewer rows than requested, the driver prints `(PARTIAL)`; the exporter reads whatever was saved.

Data Designer also **aborts early** when more than 50 % of tasks error over a window of 10. Saved batches survive that.

## 4. Inspecting results

[`scripts/inspect_synthetic_entailmentbank.py`](../scripts/inspect_synthetic_entailmentbank.py) reads a preview
`.parquet` / `.jsonl` or a `create` dataset directory. No API calls; nothing is modified. A row's **status** is
`pass`, `judge` (lowest score below `--min-judge-score`, default 4), `no-judge` (no usable scores) or
`structure` (the record failed a structural check).

```bash
.venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py summary <path> --requested 1400
.venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py show <path> --failed --below 4 --score step_validity -n 3
.venv/Scripts/python scripts/inspect_synthetic_entailmentbank.py flat <path> results.csv
```

* `summary` — status counts, structure errors, score distribution and means, which scores fail in the judge-failed
  rows, and pass rate by `n_steps`, domain and question style. `--requested N` reports how many rows are missing.
* `show` — each record as readable text with the judge's reasoning per score. Filters: `--failed`, `--passed`,
  `--id` (id substring or row number), `--domain`, `--n-steps`, `--below N` (add `--score NAME` to test one score),
  `--sample --seed S` for random rows, `-n`, `--offset`, `--full` (no truncation).
* `flat` — one row per record with only plain columns (ids, status, scores, reasoning, question, answer,
  hypothesis, proof text). Use this for tools that cannot read the nested columns of the raw parquet (a PyCharm
  parquet plugin failed on them with `NotSerializableException: com.google.gson.JsonArray`) or for Excel.
  Output is `.csv` (UTF-8 with BOM) or `.parquet`.

Rows that Data Designer dropped after exhausting retries are not in the file, so the parquet cannot count
them; compare the row count with what you requested (`--requested`) and check the run log.

## 5. Exporting

```bash
.venv/Scripts/python scripts/export_synthetic_entailmentbank.py \
    --input artifacts/entailmentbank_synth_v1 --out data/synthetic/entailmentbank_synth_v1 --rejects
```

`--input` is the dataset directory printed by `create`, or a preview `.parquet` / `.jsonl`.

The exporter rebuilds each task_1 record from the saved `skeleton` and `tree` columns, re-checks it, filters,
then derives the other tasks and validates **every** output record against `docs/schemas/*.schema.json` (it
aborts if any record fails). Output:

```
<out>/dataset/task_1/{train,dev,test}.jsonl      gold leaves only, with proof
<out>/dataset/task_2/{train,dev,test}.jsonl      25 sentences (gold + distractors), proof renumbered
<out>/dataset/task_3/{train,dev,test}.jsonl      25 retrieved sentences, ~77 % of gold leaves, proof == ""
<out>/dataset/task_3/all.jsonl
<out>/manifest.json                               counts, drop reasons, QC stats, per-record judge scores
<out>/rejects.jsonl                               only with --rejects
```

Ids are `SYN_<domain>_<hex>`. Synthetic files live in their own directory and are never mixed into the real
splits; `EntailmentBank(task=..., split=..., local_dir=<out>)` loads them.

### Exporter options

| Option | Default | Meaning |
|---|---|---|
| `--input PATH` | required | Data Designer output directory, or a `.parquet` / `.jsonl` file. |
| `--out DIR` | `data/synthetic/entailmentbank_synth_v1` | Output directory. |
| `--min-judge-score N` | 4 | **Every** judge score must be at least N (1-5), else the row is dropped. |
| `--keep-all` | off | Skip the judge filter. Structure checks and duplicate removal still apply. |
| `--rejects [PATH]` | off | Write every dropped row to a JSONL file; without a PATH, `<out>/rejects.jsonl`. |
| `--split-fractions` | `0.8,0.1,0.1` | train, dev, test; assigned by a hash of the id, so stable. |
| `--task3-dup-prob P` | 0.0 | Chance a task_3 context repeats one sentence (87 % of real contexts do). |
| `--real-data DIR` | auto-detected | Real EntailmentBank directory, used for overlap statistics. |
| `--drop-real-overlap` | off | Also drop records whose question or hypothesis text appears in the real data. |
| `--seed N` | 0 | Seeds the distractor and context choices. |
| `--limit N` | all | Only read the first N rows. |
| `--no-schema` | off | Skip JSON Schema validation. |

### What gets dropped, and the rejects file

Reasons counted in `manifest.json` under `dropped`:

| Reason | Cause |
|---|---|
| `structure/consistency` | Rebuilt record failed a structural check (missing leaf, empty hypothesis, duplicate sentence, ...). Never exportable, even with `--keep-all`. |
| `no judge scores` | The judge column is missing or unparseable. |
| `judge < N` | Lowest score below `--min-judge-score`. |
| `duplicate id/question/hypothesis` | Repeats an earlier kept row. |
| `question/hypothesis exists in real data` | Only with `--drop-real-overlap`. |
| `pool too small for 25-sentence context` | Not enough other records to build 25-sentence task_2/task_3 contexts. Only happens with very small inputs. |

With `--rejects`, each dropped row becomes one JSON line:

| Field | Contents |
|---|---|
| `reason` | One of the reasons above. |
| `record_uuid`, `id`, `domain`, `topic`, `question_style`, `n_steps` | Identification. |
| `judge_scores` | The four integer scores (`null` where missing). |
| `judge_raw` | The judge's full output, including its written reasoning per score. |
| `structure_errors` | The failed checks, for structure rejects. |
| `record` | The rebuilt task_1 record (`null` if it could not be built). |
| `tree` | The LLM's raw sentences. |
| `skeleton_text` | The proof graph in readable form. |

`--keep-all` together with `--rejects` only captures structure failures and duplicates (nothing is rejected on
judge scores). For failure analysis, use the default threshold and `--rejects`.

## 6. Retries and failures at a glance

| What fails | What happens |
|---|---|
| HTTP 500 / 502 / 503 / 504 | Retried up to 5 times by the HTTP layer (the driver adds 500). |
| LLM output that does not parse | Whole call re-run, up to `--max-restarts` (default 5). Row dropped if it still fails. |
| Row fails after all retries | Dropped by Data Designer; **it is not in the output at all**. |
| Judge score below threshold | **Nothing is retried.** The row is kept and filtered at export. |
| Empty OpenAI balance | Run aborts quickly with a quota error (below). |
| More than 50 % task errors in a window of 10 | Data Designer aborts early; saved batches are kept. |

## Known problems

**`data-designer preview|create` crash** with `Option() got an unexpected keyword argument 'click_type'` (Typer
0.29.x). `validate` works; the Python API works. Use the driver. (`data-designer validate` on Windows needs
`PYTHONUTF8=1` because of an emoji in its output; the driver sets UTF-8 itself.)

**Intermittent HTTP 500 from OpenAI** (seen on `gpt-6-luna`): Data Designer treats it as fatal for a health
check or row, although the identical request succeeds on retry. The driver widens the retry set to include 500.
This patches a private Data Designer module and prints a warning, then carries on with defaults, if that
module changes.

### Out of credits (HTTP 429)

OpenAI answers an empty balance with `429 ... "You have no credits remaining"` (`credit_balance_exhausted`).
Data Designer recognises quota errors by phrase and this wording was not among them, so the error was
classified as a rate limit and retried indefinitely: a preview logged "Provider showing degraded performance ...
rate_limit=1095, success=90" for over 30 minutes. The driver now adds the OpenAI phrases, so the run fails in
seconds with:

```
Model provider 'openai' reported insufficient credits or quota for model '...' while running health checks.
```

Add credits at https://platform.openai.com/settings/organization/billing/ and rerun. A genuine rate limit
("Rate limit reached ... tokens per min") is still retried as before. If you see the repeating "degraded
performance" warning with a different message, run `check-models` or one direct request to read the 429 body.

## What has and has not been verified

* The pure logic (skeleton, encoders against a real record, `build_record`) is covered by the offline tests.
* Two 10-record previews and the exporter ran end to end against the real API before the later options were added.
* `--rejects`, the empty-credits patch and `--buffer-size` were checked offline (exporter on a saved preview,
  classification of the 429 text, `check-models` against an empty balance, `RunConfig` values).
  `--max-restarts` was checked only up to the `RunConfig` value. **None of the four has been run in a billed
  `preview` or `create` yet**, and checkpoint writing and `--resume` have not been exercised.
