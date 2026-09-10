"""End-to-end check of the nnsight extraction path against a tiny local model.

Opt-in (downloads ~0.5 GB for gpt2, needs torch + nnsight + transformers):

    RT_MODEL_TESTS=1 python -m pytest tests/test_model_hidden_states.py -q
"""
import os

import numpy as np
import pytest

if not os.environ.get("RT_MODEL_TESTS"):
    pytest.skip("set RT_MODEL_TESTS=1 to run model-backed tests", allow_module_level=True)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
pytest.importorskip("torch")
pytest.importorskip("nnsight")

from reasoning_trajectories.config import ModelConfig
from reasoning_trajectories.extraction import ReasoningModel, layers as layer_utils

MODEL = "openai-community/gpt2"


@pytest.fixture(scope="module")
def model():
    return ReasoningModel(ModelConfig(name=MODEL, remote=False, dtype="float32",
                                      device_map="cpu", dispatch=True))


def test_hidden_states_shapes(model):
    L = model.n_layers
    h = model.hidden_size
    l3, l23 = layer_utils.resolve_layer(1 / 3, L), layer_utils.resolve_layer(2 / 3, L)
    text = "the sun rises in the east. therefore the sun does not rise in the west."
    prompt_len = len(model.tokenizer("the sun rises in the east.", add_special_tokens=True).input_ids)

    hs = model.hidden_states(text, token_layers=[l3, l23],
                             want_layer_trajectory=True, chain_start=prompt_len)

    n_chain = hs["n_positions"] - hs["chain_start"]
    assert set(hs["token_hidden"]) == {l3, l23}
    for l in (l3, l23):
        assert hs["token_hidden"][l].shape == (n_chain, h)          # (seq, hidden), no batch axis
        assert np.isfinite(hs["token_hidden"][l]).all()
    assert len(hs["layer_means"]) == L + 1
    assert hs["layer_means"][0].shape == (h,)
    assert len(hs["tokens"]) == n_chain

    # layer-l mean == mean of that layer's token hidden states (when it was saved in full)
    assert np.allclose(hs["layer_means"][l23], hs["token_hidden"][l23].mean(axis=0), atol=1e-3)
