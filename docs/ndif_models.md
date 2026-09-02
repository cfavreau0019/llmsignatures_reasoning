# NNsight / NDIF notes

## Package

`nnsight` (the "NNDIF" package) from the [ndif-team](https://github.com/ndif-team/nnsight).
The same interpretability code runs locally or remotely on the National Deep
Inference Fabric (NDIF); remote is opt-in per call with `remote=True`.

```python
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")           # or env NDIF_API_KEY, or .env
model = LanguageModel("meta-llama/Meta-Llama-3.1-70B")

with model.trace("The Eiffel Tower is in", remote=True):
    h = model.model.layers[40].output[0].save()   # residual stream after block 40
```

This repo wraps all of that in
[`extraction/model.py`](../src/reasoning_trajectories/extraction/model.py):

- residual-stream point `0` = `model.embed_tokens` output; point `k` = `model.layers[k-1]` output.
- token trajectory: `layers[l*-1].output[0][0, chain_start:, :]` saved.
- layer trajectory: `...output[0][0, chain_start:, :].mean(dim=0)` saved per layer
  (reduction done server-side to keep transfers small).

## Hosted models

The authoritative, live list is <https://nnsight.net/status/>. As documented by
NDIF it includes multiple sizes of **Llama 3.1** (8B / 70B, base and Instruct;
405B requires an access application) and **DeepSeek-R1** distill models, among
others. Any Hugging Face id also works **locally** (`remote: false`) if you have
the hardware.

Set the model in a config:

```yaml
model:
  name: meta-llama/Meta-Llama-3.1-70B
  remote: true
  api_key: null           # or env NDIF_API_KEY
```

## Credentials / environment

`reasoning_trajectories` loads a project `.env` on import
([`env.py`](../src/reasoning_trajectories/env.py)) without overriding real
environment variables. Create a git-ignored `.env` in the repo root (never
commit it):

- `NDIF_API_KEY` (fallback `NNSIGHT_API_KEY`) — passed to
  `CONFIG.set_default_api_key` when `model.remote` is true and `model.api_key`
  is not set in the config.
- `HF_TOKEN` — used by `transformers` for gated repos (all `meta-llama/*`).
- `HF_HOME` — optional cache relocation.

## Version notes

- Tested against the `nnsight` 0.5/0.6 trace API (`model.trace`, `.save()`,
  `model.generator.output` for generated ids).
- If `LanguageModel(...)` rejects `torch_dtype`, drop `model.dtype` handling or
  pass the arg your installed version expects — it is the only version-sensitive
  spot in `extraction/model.py`.
