from .base import Example, ReasoningDataset
from .entailmentbank import EntailmentBank
from .openbookqa import OpenBookQA, format_prompt


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
        return EntailmentBank(
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
    raise ValueError(f"Unknown dataset '{cfg.name}'. Add a loader in datasets/.")


__all__ = [
    "Example",
    "ReasoningDataset",
    "OpenBookQA",
    "EntailmentBank",
    "format_prompt",
    "load_dataset_from_config",
]
