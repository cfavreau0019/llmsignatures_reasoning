from .base import Example, ReasoningDataset
from .openbookqa import OpenBookQA, format_prompt


def load_dataset_from_config(cfg) -> ReasoningDataset:
    """Build a dataset from a `DatasetConfig`."""
    name = cfg.name.lower()
    if name in ("openbookqa", "obqa"):
        return OpenBookQA(
            hf_path=cfg.hf_path,
            hf_config=cfg.hf_config,
            split=cfg.split,
            limit=cfg.limit,
            include_fact=cfg.include_fact,
            shuffle_seed=cfg.shuffle_seed,
        )
    raise ValueError(f"Unknown dataset '{cfg.name}'. Add a loader in datasets/.")


__all__ = ["Example", "ReasoningDataset", "OpenBookQA", "format_prompt", "load_dataset_from_config"]
