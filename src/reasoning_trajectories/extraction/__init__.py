from . import layers
from .record import ExtractionRecord

__all__ = ["layers", "ExtractionRecord", "Extractor", "ModelPaths", "ReasoningModel"]


def __getattr__(name):  # lazy: don't import torch/nnsight unless needed
    if name in ("Extractor",):
        from .extract import Extractor

        return Extractor
    if name in ("ReasoningModel", "ModelPaths"):
        from .model import ModelPaths, ReasoningModel

        return {"ReasoningModel": ReasoningModel, "ModelPaths": ModelPaths}[name]
    raise AttributeError(name)
