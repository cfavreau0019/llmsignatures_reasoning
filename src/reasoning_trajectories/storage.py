"""On-disk layout for a run.

    {output.root}/{run_name}/
        config.yaml                  resolved config
        records/{example_id}.npz     hidden states + trajectories (one file per chain)
        chains.jsonl                 one row per chain: text, tokens, answer, prediction
        features.parquet             one row per (chain, trajectory) with FEATURE_KEYS
        manifest.json                run-level summary

`.npz` keys:
    token_layer_{L}      float{16,32}  (n_chain_pos, h)   token trajectory at layer L
    layer_trajectory     float32       (n_points, h)      depth trajectory
    layer_indices        int32         (n_points,)
    token_layers         int32         which L are present
"""
from __future__ import annotations

import json
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .extraction.record import ExtractionRecord

_STR_DTYPE = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}


class RunStore:
    def __init__(self, run_dir: str | Path, store_dtype: str = "float16"):
        self.root = Path(run_dir)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.store_dtype = _STR_DTYPE.get(store_dtype, np.float16)
        self.chains_path = self.root / "chains.jsonl"

    # ------------------------------------------------------------------ #
    def save_record(self, rec: "ExtractionRecord", *, save_hidden: bool = True, save_tokens: bool = True) -> None:
        if save_hidden:
            arrays: dict[str, np.ndarray] = {}
            present = []
            for l, tt in rec.token_trajectories.items():
                arrays[f"token_layer_{l}"] = tt.points.astype(self.store_dtype)
                present.append(l)
            arrays["token_layers"] = np.asarray(sorted(present), dtype=np.int32)
            if rec.layer_trajectory is not None:
                arrays["layer_trajectory"] = rec.layer_trajectory.points.astype(np.float32)
                arrays["layer_indices"] = rec.layer_trajectory.layer_indices.astype(np.int32)
            np.savez_compressed(self.records_dir / f"{_safe(rec.example_id)}.npz", **arrays)

        row = {
            "id": rec.example_id,
            "star_layer": rec.star_layer,
            "n_positions": rec.n_positions,
            "chain_start": rec.chain_start,
            "answer_key": rec.answer_key,
            "predicted": rec.predicted,
            "is_correct": rec.is_correct,
            "reasoning": rec.reasoning,
            "prompt": rec.prompt,
            "tokens": rec.tokens if save_tokens else None,
            "meta": rec.meta,
        }
        with self.chains_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    def load_arrays(self, example_id: str) -> dict[str, np.ndarray]:
        with np.load(self.records_dir / f"{_safe(example_id)}.npz") as z:
            return {k: z[k] for k in z.files}

    def iter_chains(self):
        if not self.chains_path.exists():
            return
        for line in self.chains_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)

    def write_features(self, rows: list[dict]) -> Path:
        import pandas as pd

        df = pd.DataFrame(rows)
        out = self.root / "features.parquet"
        df.to_parquet(out, index=False)
        df.to_csv(self.root / "features.csv", index=False)
        return out

    def write_manifest(self, manifest: dict) -> None:
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
