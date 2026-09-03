"""Typed configuration for extraction runs.

A run is fully described by a `Config` tree. Load one from YAML with
`Config.load("configs/default.yaml")` and optionally pass dotted overrides:
`Config.load(path, overrides={"model.remote": True, "dataset.limit": 4})`.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3.1-8B"
    remote: bool = False
    api_key: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    dispatch: bool = True


@dataclass
class DatasetConfig:
    name: str = "openbookqa"
    hf_path: str = "allenai/openbookqa"
    hf_config: str = "main"
    split: str = "validation"
    limit: int | None = None
    include_fact: bool = False
    shuffle_seed: int = 0
    # dataset-specific knobs (e.g. EntailmentBank: task, local_dir, reasoning_format)
    params: dict = field(default_factory=dict)


@dataclass
class ReasoningConfig:
    source: str = "generate"  # generate | dataset_field | precomputed
    max_new_tokens: int = 256
    temperature: float = 0.0
    stop_on_answer: bool = True
    precomputed_path: str | None = None


@dataclass
class ExtractionConfig:
    star_layer: str | int | float = "two_thirds"
    extra_token_layers: list = field(default_factory=list)
    layer_trajectory: bool = True
    token_trajectory: bool = True
    chain_span: str = "completion"  # completion | full
    store_dtype: str = "float16"
    batch_size: int = 1


@dataclass
class GeometryConfig:
    step_rel_tol: float = 1e-6
    step_abs_tol: float = 1e-12
    plane_sin2_tol: float = 1e-10


@dataclass
class OutputConfig:
    root: str = "data/runs"
    save_hidden_states: bool = True
    save_tokens: bool = True


_SECTIONS = {
    "model": ModelConfig,
    "dataset": DatasetConfig,
    "reasoning": ReasoningConfig,
    "extraction": ExtractionConfig,
    "geometry": GeometryConfig,
    "output": OutputConfig,
}


@dataclass
class Config:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output.root) / self.run_name

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        kwargs: dict[str, Any] = {}
        if "run_name" in raw:
            kwargs["run_name"] = raw["run_name"]
        for key, section_cls in _SECTIONS.items():
            if key in raw and raw[key] is not None:
                allowed = {f.name for f in fields(section_cls)}
                bad = set(raw[key]) - allowed
                if bad:
                    raise ValueError(f"Unknown keys in config section '{key}': {sorted(bad)}")
                kwargs[key] = section_cls(**raw[key])
        cfg = cls(**kwargs)
        for dotted, value in (overrides or {}).items():
            _set_dotted(cfg, dotted, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _to_dict(obj):
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _set_dotted(obj, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"No config field '{dotted}'")
    setattr(obj, parts[-1], value)
