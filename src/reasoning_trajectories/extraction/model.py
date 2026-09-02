"""Thin wrapper around an ``nnsight`` model (local or NDIF-remote).

Responsibilities
----------------
* build the ``LanguageModel`` and read its depth / width;
* generate a reasoning chain for a prompt;
* run one trace over ``prompt + chain`` and return hidden states for the
  targeted residual-stream points only.

Residual-stream point convention (see ``extraction/layers.py``):
    point 0        -> token-embedding output
    point k (>=1)  -> output of transformer block k  (module ``layers[k-1]``)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ModelPaths:
    """Attribute paths for the residual stream, per architecture family."""
    blocks: str          # module list of transformer blocks
    embeddings: str       # token-embedding module
    block_output_is_tuple: bool = True

    @staticmethod
    def for_model(hf_name: str) -> "ModelPaths":
        n = hf_name.lower()
        if "gpt2" in n or "gpt-2" in n:
            return ModelPaths(blocks="transformer.h", embeddings="transformer.wte")
        if "gptneox" in n or "pythia" in n:
            return ModelPaths(blocks="gpt_neox.layers", embeddings="gpt_neox.embed_in")
        # llama / mistral / qwen2 / gemma / deepseek-r1-distill ...
        return ModelPaths(blocks="model.layers", embeddings="model.embed_tokens")


def _getattr_path(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class ReasoningModel:
    def __init__(self, cfg):
        from nnsight import LanguageModel

        self.cfg = cfg
        self.remote = bool(cfg.remote)
        if self.remote:
            self._configure_api_key(cfg.api_key)

        dtype = getattr(torch, cfg.dtype) if isinstance(cfg.dtype, str) else cfg.dtype
        kwargs = {"torch_dtype": dtype}
        if not self.remote:
            kwargs.update(device_map=cfg.device_map, dispatch=cfg.dispatch)
        self.lm = LanguageModel(cfg.name, **kwargs)

        self.paths = ModelPaths.for_model(cfg.name)
        self.n_layers = int(self.lm.config.num_hidden_layers)
        self.hidden_size = int(self.lm.config.hidden_size)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _configure_api_key(explicit: str | None) -> None:
        key = explicit or os.environ.get("NDIF_API_KEY") or os.environ.get("NNSIGHT_API_KEY")
        if not key:
            raise RuntimeError(
                "Remote execution requested but no API key found. Set model.api_key "
                "in the config or the NDIF_API_KEY environment variable. "
                "Get a key at https://login.ndif.us"
            )
        from nnsight import CONFIG

        CONFIG.set_default_api_key(key)

    # ------------------------------------------------------------------ #
    @property
    def tokenizer(self):
        return self.lm.tokenizer

    def _point_proxy(self, point_index: int):
        """nnsight proxy for the hidden state at a residual-stream point."""
        if point_index == 0:
            return _getattr_path(self.lm, self.paths.embeddings).output
        block = _getattr_path(self.lm, self.paths.blocks)[point_index - 1]
        return block.output[0] if self.paths.block_output_is_tuple else block.output

    # ------------------------------------------------------------------ #
    def generate_reasoning(self, prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
        """Return the generated continuation (text after the prompt)."""
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with self.lm.generate(prompt, remote=self.remote, **gen_kwargs):
            out = self.lm.generator.output.save()

        seq = _to_numpy(out)[0]
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True).input_ids)
        completion_ids = seq[prompt_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    def hidden_states(
        self,
        text: str,
        *,
        token_layers: list[int],
        want_layer_trajectory: bool,
        chain_start: int = 0,
    ) -> dict:
        """One trace over ``text``.

        Returns
        -------
        dict with
          ``token_hidden``  : {layer -> np.ndarray (n_chain_pos, h)}  (chain positions only)
          ``layer_means``   : {point -> np.ndarray (h,)}  mean over chain positions, every point
                              (only if ``want_layer_trajectory``)
          ``tokens``        : list[str]  (chain positions only)
          ``n_positions``   : int  total sequence length
        """
        enc = self.tokenizer(text, add_special_tokens=True)
        n_pos = len(enc.input_ids)
        chain_start = max(0, min(chain_start, n_pos - 1))

        token_layers = sorted(set(token_layers))
        points_for_means = list(range(self.n_layers + 1)) if want_layer_trajectory else []

        with self.lm.trace(text, remote=self.remote):
            tok_saves = {
                l: self._point_proxy(l)[0, chain_start:, :].save() for l in token_layers
            }
            mean_saves = {
                p: self._point_proxy(p)[0, chain_start:, :].mean(dim=0).save()
                for p in points_for_means
            }

        token_hidden = {l: _to_numpy(v) for l, v in tok_saves.items()}
        layer_means = {p: _to_numpy(v).reshape(-1) for p, v in mean_saves.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(enc.input_ids[chain_start:])
        return {
            "token_hidden": token_hidden,
            "layer_means": layer_means,
            "tokens": tokens,
            "n_positions": n_pos,
            "chain_start": chain_start,
        }


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "value"):          # unresolved nnsight proxy handle
        x = x.value
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x)
