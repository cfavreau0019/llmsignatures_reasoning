import numpy as np
import pytest

from reasoning_trajectories.config import Config
from reasoning_trajectories.extraction import layers


def test_two_thirds_matches_note():
    # Llama-3.1-8B has 32 blocks -> l* = round(2/3 * 32) = 21
    assert layers.two_thirds_layer(32) == 21
    assert layers.resolve_layer("two_thirds", 32) == 21


def test_resolve_layer_forms():
    assert layers.resolve_layer(0, 32) == 0
    assert layers.resolve_layer(-1, 32) == 32          # final residual stream
    assert layers.resolve_layer(0.5, 32) == 16
    assert layers.resolve_layer("last", 32) == 32
    assert layers.resolve_layer("embeddings", 32) == 0
    with pytest.raises(ValueError):
        layers.resolve_layer(99, 32)


def test_resolve_layers_list_and_slice():
    assert layers.resolve_layers([0, 0.5, "two_thirds", -1], 30) == sorted({0, 15, 20, 30})
    assert layers.resolve_layers("0:5", 30) == [0, 1, 2, 3, 4]


def test_config_roundtrip(tmp_path):
    cfg = Config.load("configs/default.yaml", overrides={"model.remote": True, "dataset.limit": 3})
    assert cfg.model.remote is True
    assert cfg.dataset.limit == 3
    p = tmp_path / "c.yaml"
    cfg.save(p)
    again = Config.load(p)
    assert again.to_dict() == cfg.to_dict()


def test_config_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  nonsense: 1\n")
    with pytest.raises(ValueError):
        Config.load(bad)
