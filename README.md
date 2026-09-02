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
| `src/reasoning_trajectories/datasets/` | dataset loaders → `Example` (OpenBookQA included; add more here) |
| `src/reasoning_trajectories/reasoning/` | generate / parse the CoT text, score the final answer |
| `src/reasoning_trajectories/extraction/layers.py` | resolve layer specs (`two_thirds`, floats, ints, slices) |
| `src/reasoning_trajectories/extraction/model.py` | NNsight wrapper: `.generate`, `.trace`, targeted hidden states |
| `src/reasoning_trajectories/extraction/extract.py` | per-example: text → trajectories |
| `src/reasoning_trajectories/trajectories/` | **the geometry** — `geometry.py`, `guards.py`, `features.py`, `trajectory.py` |
| `src/reasoning_trajectories/storage.py` | on-disk run layout (`.npz` + `chains.jsonl` + `features.parquet`) |
| `src/reasoning_trajectories/pipeline.py` | end-to-end `run_pipeline(cfg)` |
| **`studies/hidden_state_trajectories/`** | **analysis of the trajectories** — profiles, outcome comparisons, plots |
| `scripts/run_extraction.py` | CLI for a run |
| `scripts/recompute_features.py` | recompute features from stored hidden states (no model) |
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

For **remote** (NDIF-hosted) models get an API key at <https://login.ndif.us> and
either set `model.api_key` in the config or export `NDIF_API_KEY`.
Hosted model list: <https://nnsight.net/status/>. See
[`docs/ndif_models.md`](docs/ndif_models.md).

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
```

## Adding another dataset

Write a loader in `src/reasoning_trajectories/datasets/` that yields `Example`
objects and register it in `datasets/__init__.py::load_dataset_from_config`.
Nothing else changes.

## Tests

```bash
python -m pytest -q
```
