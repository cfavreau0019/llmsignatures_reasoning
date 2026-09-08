from .base import Example, ReasoningDataset
from .entailmentbank import EntailmentBank
from .openbookqa import OpenBookQA, format_prompt
from .perturbations import CYCLE_DEFAULT, PairedCorruption


def load_dataset_from_config(cfg) -> ReasoningDataset:
    """Build a dataset from a `DatasetConfig`."""
    name = cfg.name.lower().replace("-", "").replace("_", "")
    params = dict(getattr(cfg, "params", None) or {})

    if name in ("openbookqa", "obqa"):
        return OpenBookQA(
            hf_path=cfg.hf_path,
            hf_config=cfg.hf_config,
            split=cfg.split,
            limit=cfg.limit,
            include_fact=cfg.include_fact,
            shuffle_seed=cfg.shuffle_seed,
        )
    if name in ("entailmentbank", "eb"):
        ds = EntailmentBank(
            task=params.get("task", 1),
            split=cfg.split,
            local_dir=params.get("local_dir"),
            data_files=params.get("data_files"),
            hf_path=params.get("hf_path"),
            limit=cfg.limit,
            shuffle_seed=cfg.shuffle_seed,
            reasoning_format=params.get("reasoning_format", "natural"),
            use_question=params.get("use_question", False),
        )
        return _maybe_perturb(ds, params)
    raise ValueError(f"Unknown dataset '{cfg.name}'. Add a loader in datasets/.")


def _maybe_perturb(ds, params: dict):
    """Wrap `ds` in PairedCorruption when dataset.params.perturbation is set.

    params.perturbation: true, or a dict with keys
        enabled (default true), min_steps (3), seed (0), emit ("both"),
        cycle (list of corruption kinds).
    """
    p = params.get("perturbation")
    if not p:
        return ds
    p = {} if p is True else dict(p)
    if not p.get("enabled", True):
        return ds
    return PairedCorruption(
        ds,
        min_steps=p.get("min_steps", 3),
        seed=p.get("seed", 0),
        cycle=tuple(p.get("cycle") or CYCLE_DEFAULT),
        emit=p.get("emit", "both"),
        reasoning_format=params.get("reasoning_format", "natural"),
    )


__all__ = [
    "Example",
    "ReasoningDataset",
    "OpenBookQA",
    "EntailmentBank",
    "PairedCorruption",
    "CYCLE_DEFAULT",
    "format_prompt",
    "load_dataset_from_config",
]
