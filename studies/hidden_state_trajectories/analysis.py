"""Analysis helpers for hidden-state trajectory features.

Everything here reads a completed run directory
(``data/runs/<run_name>/``) produced by the extraction pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reasoning_trajectories.config import Config  # noqa: E402
from reasoning_trajectories.storage import RunStore  # noqa: E402
from reasoning_trajectories.trajectories import Guards, curvature, torsion  # noqa: E402
from reasoning_trajectories.trajectories.features import FEATURE_KEYS  # noqa: E402

SCALAR_FEATURES = [k for k in FEATURE_KEYS if not k.endswith("_frac") and k != "n_points"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.yaml")
    store = RunStore(run_dir, store_dtype=cfg.extraction.store_dtype)
    features = pd.read_parquet(run_dir / "features.parquet")
    return cfg, store, features


# --------------------------------------------------------------------------- #
# aggregate comparisons
# --------------------------------------------------------------------------- #
def describe_by_outcome(features: pd.DataFrame, trajectory: str = "token") -> pd.DataFrame:
    """Mean +/- std of each scalar feature, split by answer correctness."""
    sub = features[features["trajectory"] == trajectory]
    sub = sub[sub["is_correct"].notna()]
    g = sub.groupby("is_correct")[SCALAR_FEATURES]
    return pd.concat({"mean": g.mean(), "std": g.std(), "n": g.size().to_frame("n")}, axis=1)


def outcome_effect_sizes(features: pd.DataFrame, trajectory: str = "token") -> pd.DataFrame:
    """Cohen's d (correct - incorrect) and Welch t-style z for each feature."""
    sub = features[(features["trajectory"] == trajectory) & features["is_correct"].notna()]
    a = sub[sub["is_correct"]][SCALAR_FEATURES]
    b = sub[~sub["is_correct"]][SCALAR_FEATURES]
    if len(a) < 2 or len(b) < 2:
        return pd.DataFrame(index=SCALAR_FEATURES)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    d = (a.mean() - b.mean()) / pooled.replace(0, np.nan)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se.replace(0, np.nan)
    return pd.DataFrame({"cohen_d": d, "welch_z": z, "n_correct": len(a), "n_incorrect": len(b)})


def correlation_matrix(features: pd.DataFrame, trajectory: str = "token") -> pd.DataFrame:
    sub = features[features["trajectory"] == trajectory]
    return sub[SCALAR_FEATURES].corr()


# --------------------------------------------------------------------------- #
# valid vs. corrupted chains (PairedCorruption runs)
# --------------------------------------------------------------------------- #
def describe_by_condition(features: pd.DataFrame, trajectory: str = "token") -> pd.DataFrame:
    """Mean / std of each scalar feature, split by `condition` (valid|corrupted)."""
    sub = features[(features["trajectory"] == trajectory) & features["condition"].notna()]
    g = sub.groupby("condition")[SCALAR_FEATURES]
    return pd.concat({"mean": g.mean(), "std": g.std(), "n": g.size().to_frame("n")}, axis=1)


def paired_feature_deltas(features: pd.DataFrame, trajectory: str = "token",
                          by_corruption: bool = False) -> pd.DataFrame:
    """Per-pair corrupted-minus-valid deltas, joined on `pair_id`.

    Returns mean delta, its std, a paired-t statistic (mean / SE) and Cohen's d_z
    for each feature -- optionally broken out by `corruption_kind`.
    """
    sub = features[(features["trajectory"] == trajectory) & features["condition"].notna()]
    v = sub[sub["condition"] == "valid"].set_index("pair_id")
    c = sub[sub["condition"] == "corrupted"].set_index("pair_id")
    pairs = v.index.intersection(c.index)
    if len(pairs) < 2:
        return pd.DataFrame(index=SCALAR_FEATURES)

    delta = c.loc[pairs, SCALAR_FEATURES] - v.loc[pairs, SCALAR_FEATURES]
    kind = c.loc[pairs, "corruption_kind"]

    def _stats(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        m, sd = d.mean(), d.std(ddof=1)
        se = sd / np.sqrt(n)
        return pd.DataFrame({"n_pairs": n, "mean_delta": m, "std_delta": sd,
                             "paired_t": m / se.replace(0, np.nan),
                             "cohen_dz": m / sd.replace(0, np.nan)})

    if not by_corruption:
        return _stats(delta)
    return pd.concat({k: _stats(delta[kind == k]) for k in sorted(kind.unique())}, names=["corruption"])


# --------------------------------------------------------------------------- #
# per-vertex profiles along one chain
# --------------------------------------------------------------------------- #
def vertex_profiles(store: RunStore, example_id: str, cfg: Config, layer: int | None = None) -> dict:
    """Curvature / torsion as a function of position along the token trajectory."""
    arr = store.load_arrays(example_id)
    guards = Guards.from_config(cfg.geometry)
    if layer is None:
        layer = int(arr["token_layers"][0])
    X = arr[f"token_layer_{layer}"].astype(np.float64)
    kappa, kappa_valid = curvature(X, guards)
    tau, tau_valid = torsion(X, guards)
    return {
        "layer": layer,
        "n_points": X.shape[0],
        "curvature": kappa,
        "curvature_valid": kappa_valid,
        "torsion": tau,
        "torsion_valid": tau_valid,
    }


def layer_trajectory_profile(store: RunStore, example_id: str, cfg: Config) -> dict:
    """Curvature / torsion vs. depth for the layer trajectory."""
    arr = store.load_arrays(example_id)
    guards = Guards.from_config(cfg.geometry)
    X = arr["layer_trajectory"].astype(np.float64)
    kappa, kv = curvature(X, guards)
    tau, tv = torsion(X, guards)
    return {
        "layer_indices": arr["layer_indices"],
        "step_norms": np.linalg.norm(np.diff(X, axis=0), axis=1),
        "curvature": kappa, "curvature_valid": kv,
        "torsion": tau, "torsion_valid": tv,
    }


# --------------------------------------------------------------------------- #
# plots  (matplotlib; each returns the saved path)
# --------------------------------------------------------------------------- #
def plot_feature_distributions(features: pd.DataFrame, out_path: str | Path,
                               trajectory: str = "token") -> Path:
    import matplotlib.pyplot as plt

    sub = features[features["trajectory"] == trajectory]
    feats = [f for f in SCALAR_FEATURES if sub[f].notna().any()]
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    has_outcome = sub["is_correct"].notna().any()
    for ax, f in zip(axes, feats):
        if has_outcome:
            for val, label, color in [(True, "correct", "tab:green"), (False, "incorrect", "tab:red")]:
                d = sub[sub["is_correct"] == val][f].dropna()
                if len(d):
                    ax.hist(d, bins=20, alpha=0.55, label=label, color=color)
            ax.legend(fontsize=7)
        else:
            ax.hist(sub[f].dropna(), bins=20, color="tab:blue")
        ax.set_title(f, fontsize=9)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"{trajectory} trajectory features")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_chain_profile(store: RunStore, example_id: str, cfg: Config, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    prof = vertex_profiles(store, example_id, cfg)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(np.where(prof["curvature_valid"], prof["curvature"], np.nan), color="tab:blue")
    axes[0].set_ylabel(r"$\kappa_i$")
    axes[1].plot(np.where(prof["torsion_valid"], prof["torsion"], np.nan), color="tab:orange")
    axes[1].set_ylabel(r"$|\tau_i|$")
    axes[1].set_xlabel("interior vertex index along chain")
    fig.suptitle(f"{example_id} — token trajectory @ layer {prof['layer']}")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_trajectory_pca(store: RunStore, example_id: str, out_path: str | Path,
                        which: str = "token", layer: int | None = None) -> Path:
    """VISUALISATION ONLY. 3-D PCA of one trajectory — not used for any metric."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    arr = store.load_arrays(example_id)
    if which == "layer":
        X = arr["layer_trajectory"].astype(np.float64)
    else:
        layer = layer if layer is not None else int(arr["token_layers"][0])
        X = arr[f"token_layer_{layer}"].astype(np.float64)
    Y = PCA(n_components=3).fit_transform(X)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "-o", ms=3, lw=1)
    ax.scatter(*Y[0], c="green", s=40, label="start")
    ax.scatter(*Y[-1], c="red", s=40, label="end")
    ax.legend()
    ax.set_title(f"{example_id} — {which} trajectory (PCA, viz only)")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
