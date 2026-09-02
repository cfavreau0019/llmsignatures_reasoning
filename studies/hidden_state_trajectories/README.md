# Hidden-state trajectory study

This folder is the analysis layer that sits on top of the extraction pipeline.
It turns the stored trajectories into the curve-geometry quantities from the
reference note and studies how they vary across reasoning chains (e.g. correct
vs. incorrect answers on OpenBookQA).

## The two trajectories

For a transformer with `L` layers and hidden width `h`, every token becomes one
`h`-vector per layer. For each reasoning chain we build two ordered point lists.

| Trajectory | Follows | Points | Built by |
|---|---|---|---|
| **Token** `X_tok = (x_1..x_n)` | the text | hidden state of every chain-token position at a fixed layer `l*` | `trajectories.build_token_trajectory` |
| **Layer** `X_lay = (xbar_0..xbar_L)` | depth | per-layer mean over chain-token positions | `trajectories.build_layer_trajectory_from_means` |

`l*` is fixed **before any experiment** at two thirds of network depth
(`extraction.layers.two_thirds_layer`), consistent with Marks & Tegmark (2023).
All geometry is computed in the full dimension `h` (float64); no PCA/UMAP — the
quantities depend only on pairwise distances and inner products. PCA appears in
`analysis.py` *only* for eyeballing a single trajectory, never for a metric.

## Quantities (code ↔ formula)

Step vectors `Δ_i = x_{i+1} − x_i`. Implemented in
[`trajectories/geometry.py`](../../src/reasoning_trajectories/trajectories/geometry.py).

| Quantity | Symbol | Formula | Function |
|---|---|---|---|
| Path length | `S` | `Σ ‖Δ_i‖` | `path_length` |
| Dirichlet energy | `E` | `½ Σ ‖Δ_i‖²` | `dirichlet_energy` |
| Straightness | `σ` | `‖x_n − x_1‖ / S` (1 = straight) | `straightness` |
| Curvature | `κ_i` | `θ_i / s̄_i`, `θ_i = arccos(⟨Δ_{i−1},Δ_i⟩ / (‖Δ_{i−1}‖‖Δ_i‖))`, `s̄_i = ½(‖Δ_{i−1}‖+‖Δ_i‖)` | `curvature` |
| Torsion | `\|τ_i\|` | `sqrt(det G(v₁,v₂,v₃)) / ‖v₁∧v₂‖²` with `v₁=½(x_{i+1}−x_{i−1})`, `v₂=x_{i+1}−2x_i+x_{i−1}`, `v₃=x_{i+2}−3x_{i+1}+3x_i−x_{i−1}`, `‖v₁∧v₂‖² = ‖v₁‖²‖v₂‖² − ⟨v₁,v₂⟩²` | `torsion` |

**Numerical guards** ([`guards.py`](../../src/reasoning_trajectories/trajectories/guards.py)):
a vertex is dropped from `κ` when an adjacent step vanishes
(`‖Δ‖ ≤ step_rel_tol · median‖Δ‖`), and from `|τ|` when the first two difference
vectors are near-parallel (`‖v₁∧v₂‖² / (‖v₁‖²‖v₂‖²) ≤ plane_sin2_tol`). The
excluded fraction is recorded per trajectory as
`curvature_excluded_frac` / `torsion_excluded_frac` and is expected < 1e-3.

**Per-chain feature vector** ([`features.py`](../../src/reasoning_trajectories/trajectories/features.py)):
`S, E, σ` plus mean / max / std of `{κ_i}` and of `{|τ_i|}`.

## Files

- `analysis.py` — load `features.parquet`, aggregate by outcome, per-vertex
  profiles along a chain, distribution and PCA plots.
- `run_study.py` — run a small extraction (if needed) and emit `report/`.

## Reproduce

```bash
pip install -r requirements.txt
python studies/hidden_state_trajectories/run_study.py --config configs/default.yaml
```
