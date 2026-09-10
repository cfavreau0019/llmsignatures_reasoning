# reasoning-trajectories

Extract and study the **hidden-state trajectories** of LLM reasoning chains.

The pipeline takes a multiple-choice reasoning dataset (OpenBookQA to start),
has an [NNsight](https://nnsight.net) model — local or hosted on
[NDIF](https://nnsight.net/status/) — produce a chain-of-thought answer, pulls
hidden states from **targeted** residual-stream layers only, and turns each
chain into two trajectories whose curve geometry (path length, Dirichlet energy,
straightness, curvature, torsion) is summarised into a feature table.

```
dataset ──▶ reasoning chain ──▶ targeted hidden states ──▶ trajectories ──▶ curve-geometry features
         (nnsight .generate)   (nnsight .trace, l* only)   token & layer      S, E, σ, κ, |τ|
```

## Layout

| Path | Purpose |
|---|---|
| `src/reasoning_trajectories/config.py` | typed run config (`configs/*.yaml`) |
| `src/reasoning_trajectories/datasets/` | dataset loaders → `Example` (OpenBookQA, EntailmentBank; `perturbations.py` = paired valid/corrupted chains) |
| `src/reasoning_trajectories/reasoning/` | `chain.py` = `ReasoningChain` (one chain: text + steps + `.corrupt()`); `generate.py` = obtain / score the chain |
| `src/reasoning_trajectories/extraction/layers.py` | resolve layer specs (`two_thirds`, floats, ints, slices) |
| `src/reasoning_trajectories/extraction/model.py` | NNsight wrapper: `.generate`, `.trace`, targeted hidden states |
| `src/reasoning_trajectories/extraction/extract.py` | per-example: text → trajectories |
| `src/reasoning_trajectories/trajectories/` | **the geometry** — `geometry.py`, `guards.py`, `features.py`, `trajectory.py` |
| `src/reasoning_trajectories/storage.py` | on-disk run layout (`.npz` + `chains.jsonl` + `features.parquet`) |
| `src/reasoning_trajectories/pipeline.py` | end-to-end `run_pipeline(cfg)` |
| **`studies/hidden_state_trajectories/`** | **analysis of the trajectories** — profiles, outcome comparisons, plots |
| `scripts/run_extraction.py` | CLI for a run |
| `scripts/recompute_features.py` | recompute features from stored hidden states (no model) |
| `notebooks/` | `explore_*` dataset walk-throughs; `corrupt_and_probe_entailmentbank.ipynb` — pick chains, corrupt them, run a model, read out geometry + hidden states at `L/3` & `2L/3` |
| `src/reasoning_trajectories/env.py` | optional `.env` loading (auto-run on package import) |
| `tests/` | analytic geometry checks (circle, helix, straight line) + plumbing |

## Which layers get extracted

Extracting every layer for every token is wasteful, so the config targets:

- **`extraction.star_layer`** — the anchor layer `l*` for the **token trajectory**.
  Default `two_thirds` → `round(2·L/3)` (Llama-3.1-8B: layer 21), fixed before
  experiments, consistent with Marks & Tegmark (2023).
- **`extraction.extra_token_layers`** — any extra anchor layers to also store
  token trajectories for (e.g. `[0.5, -1]`).
- **`extraction.layer_trajectory`** — the **layer trajectory** needs a point per
  layer, but only the *mean over chain positions*; that reduction is done inside
  the `.trace` so only one `h`-vector per layer crosses the wire (matters for
  remote runs).

## Trajectories & quantities

Full definitions and the code↔formula table:
[`studies/hidden_state_trajectories/README.md`](studies/hidden_state_trajectories/README.md).

- **Token trajectory** `X_tok = (x_1,…,x_n)` — hidden state of every reasoning-chain
  token at layer `l*`.
- **Layer trajectory** `X_lay = (x̄_0,…,x̄_L)` — per-layer mean over chain tokens.
- Global shape: `S` (path length), `E` (Dirichlet energy), `σ` (straightness).
- `κ_i` discrete curvature, `|τ_i|` unsigned discrete torsion (4-point estimate).
- Numerical guards drop vanishing-step and degenerate-plane vertices; the
  excluded fraction is recorded (expect < 1e-3).
- Per chain: `S, E, σ` + mean/max/std of `{κ_i}` and `{|τ_i|}`.

All geometry is in the full hidden dimension `h`, `float64`. No PCA/UMAP for any
metric (only for eyeballing a single chain in `analysis.plot_trajectory_pca`).

## Setup

```bash
pip install -r requirements.txt
pip install -e .            # optional: puts reasoning_trajectories on the path
```

### Credentials

Create a `.env` in the repo root (git-ignored — never commit it) with the
variables below. It is loaded automatically when `reasoning_trajectories` is
imported (via `python-dotenv` if installed, otherwise a small built-in parser).
Real environment variables always take precedence over `.env`.

```dotenv
NDIF_API_KEY=...          # only for model.remote: true
HF_TOKEN=...              # only for gated models (meta-llama/*)
```

| Variable | When needed | Read by |
|---|---|---|
| `NDIF_API_KEY` (or `NNSIGHT_API_KEY`) | `model.remote: true` | `extraction/model.py` (after `model.api_key` in the config) |
| `HF_TOKEN` | gated models such as `meta-llama/*` | `transformers` / `huggingface_hub` |
| `HF_HOME` | optional, relocate HF cache | `huggingface_hub` |

Get an NDIF key at <https://login.ndif.us>; hosted model list at
<https://nnsight.net/status/>. See [`docs/ndif_models.md`](docs/ndif_models.md).

## Run

```bash
# local, small smoke run (default config: Llama-3.1-8B, OBQA validation, 16 items)
python scripts/run_extraction.py --config configs/default.yaml

# NDIF-remote, larger model
python scripts/run_extraction.py --config configs/llama70b_remote.yaml \
    --set model.api_key=$NDIF_API_KEY --set dataset.limit=200

# study: comparisons + plots into data/runs/<run_name>/report/
python studies/hidden_state_trajectories/run_study.py --config configs/default.yaml

# iterate on geometry/guards without re-running the model
python scripts/recompute_features.py --run data/runs/obqa_llama31_8b_smoke \
    --set geometry.plane_sin2_tol=1e-8

# EntailmentBank: geometry of gold, human-authored entailment chains (no generation)
#   1. download entailment_trees_emnlp2021_data_v3.zip from
#      https://allenai.org/data/entailmentbank and unzip into data/raw/
#   2. run:
python scripts/run_extraction.py --config configs/entailmentbank.yaml
```

## Datasets

| Dataset | `dataset.name` | Reasoning source | Notes |
|---|---|---|---|
| OpenBookQA | `openbookqa` | `generate` | 4-way MC; no gold reasoning — the model writes the chain |
| EntailmentBank | `entailmentbank` | `dataset_field` | ships gold entailment trees; `params.task` 1/2/3, `params.reasoning_format` = `natural`/`structured`/`raw`. Put the unzipped v3 release at `data/raw/entailment_trees_emnlp2021_data_v3/` (auto-detected) or set `params.local_dir`. Task 3 has no stored proof — the loader borrows the matching task-1 tree. |

### Paired corruptions (valid vs. broken reasoning)

Set `dataset.params.perturbation` on an EntailmentBank run and every gold tree
with ≥ `min_steps` (default 3) steps is emitted **twice** — unchanged (`valid`)
and with one corruption (`corrupted`) — with the corruption type cycled
round-robin (seed 0) over:

| kind | what breaks |
|---|---|
| `step_shuffle` | derivation **order** (guaranteed non-identity permutation; <3-step chains escalate to a same-tree premise swap) |
| `premise_substitution_other_tree` | one step's **premise** ← a sentence from a different tree |
| `conclusion_substitution` | one step's **derived conclusion** ← a sentence from another tree; propagates to any later step that used it |

`features.parquet` then carries `condition`, `pair_id`, `corruption_kind`.
Config: [`configs/entailmentbank_corruptions.yaml`](configs/entailmentbank_corruptions.yaml).
Analysis: `analysis.describe_by_condition` / `analysis.paired_feature_deltas` in
[`studies/hidden_state_trajectories/`](studies/hidden_state_trajectories/analysis.py).

```bash
python scripts/run_extraction.py --config configs/entailmentbank_corruptions.yaml
```

Under the hood the corruptions are methods on `ReasoningChain`, usable on a
single chain with no dataset:

```python
from reasoning_trajectories import ReasoningChain
from reasoning_trajectories.reasoning import DonorPool

ch = ReasoningChain.from_entailmentbank(record)     # or .from_steps(...) / .from_text(...)
ch.premises, ch.hypothesis, ch.as_format("raw").text
bad = ch.corrupt("step_shuffle", seed=0)            # -> new chain; ch untouched
bad.meta["corruption"]                              # what changed
```

`premise_substitution_other_tree` / `conclusion_substitution` take
`donors=DonorPool(other_chains)`. To pick chains, corrupt them, and probe a
model interactively, see
[`notebooks/corrupt_and_probe_entailmentbank.ipynb`](notebooks/corrupt_and_probe_entailmentbank.ipynb).

### Adding another dataset

Write a loader in `src/reasoning_trajectories/datasets/` that yields `Example`
objects and register it in `datasets/__init__.py::load_dataset_from_config`.
Dataset-specific options go under `dataset.params` (a free-form dict). Nothing
else changes.

## Tests

```bash
python -m pytest -q
RT_MODEL_TESTS=1 python -m pytest tests/test_model_hidden_states.py -q   # opt-in: runs a local gpt2
```
