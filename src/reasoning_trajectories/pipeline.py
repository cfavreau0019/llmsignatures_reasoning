"""End-to-end run: dataset -> reasoning chains -> targeted hidden states ->
trajectories -> curve-geometry features on disk.

    from reasoning_trajectories import Config, run_pipeline
    run_pipeline(Config.load("configs/default.yaml"))
"""
from __future__ import annotations

import time

from tqdm import tqdm

from .config import Config
from .datasets import load_dataset_from_config
from .extraction import Extractor, ReasoningModel
from .reasoning import ReasoningProvider, parse_answer
from .storage import RunStore
from .trajectories import Guards, summarize_trajectory


def run_pipeline(cfg: Config) -> dict:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    dataset = load_dataset_from_config(cfg.dataset)
    model = ReasoningModel(cfg.model)
    provider = ReasoningProvider(cfg.reasoning, model=model)
    extractor = Extractor(model, cfg.extraction)
    store = RunStore(run_dir, store_dtype=cfg.extraction.store_dtype)
    guards = Guards.from_config(cfg.geometry)

    feature_rows: list[dict] = []
    n_done = n_correct = 0
    t0 = time.time()

    for ex in tqdm(dataset, total=len(dataset), desc=cfg.run_name):
        reasoning = provider.reasoning_for(ex)
        predicted = parse_answer(reasoning, ex.choice_labels)
        rec = extractor.extract(ex, reasoning, predicted=predicted)
        store.save_record(
            rec,
            save_hidden=cfg.output.save_hidden_states,
            save_tokens=cfg.output.save_tokens,
        )

        base = {
            "id": rec.example_id,
            "answer_key": rec.answer_key,
            "predicted": rec.predicted,
            "is_correct": rec.is_correct,
            "n_chain_tokens": len(rec.tokens),
        }
        for l, tt in rec.token_trajectories.items():
            tag = "token" if l == rec.star_layer else f"token_L{l}"
            feature_rows.append({**base, "trajectory": tag, "layer": l,
                                 **summarize_trajectory(tt.points, guards)})
        if rec.layer_trajectory is not None:
            feature_rows.append({**base, "trajectory": "layer", "layer": -1,
                                 **summarize_trajectory(rec.layer_trajectory.points, guards)})

        n_done += 1
        n_correct += int(bool(rec.is_correct))

    store.write_features(feature_rows)
    manifest = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "remote": cfg.model.remote,
        "dataset": cfg.dataset.name,
        "split": cfg.dataset.split,
        "n_examples": n_done,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_done) if n_done else None,
        "star_layer": extractor.star_layer,
        "extra_token_layers": extractor.extra_layers,
        "n_layers": model.n_layers,
        "hidden_size": model.hidden_size,
        "seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
    }
    store.write_manifest(manifest)
    return manifest
