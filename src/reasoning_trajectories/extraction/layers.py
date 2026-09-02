"""Resolve human-friendly layer specifications to concrete indices.

The residual stream has ``n_layers + 1`` readable points: index 0 is the
token-embedding output, index ``k`` (1..n_layers) is the output of block ``k``.
``n_layers`` is the model's ``num_hidden_layers``.

Accepted spec forms
-------------------
  int              -> that index (negative counts from the end: -1 == n_layers)
  float in [0, 1]  -> round(frac * n_layers)
  "two_thirds"     -> round(2/3 * n_layers)   (the note's l*, cf. Marks & Tegmark 2023)
  "half", "last", "first", "embeddings"
  "a:b" / "a:b:s"  -> a slice, expanded to a list
  list/tuple       -> each element resolved and de-duplicated, sorted
"""
from __future__ import annotations

from typing import Iterable

_KEYWORDS = {
    "embeddings": lambda L: 0,
    "first": lambda L: 0,
    "last": lambda L: L,
    "half": lambda L: round(L / 2),
    "two_thirds": lambda L: round(2 * L / 3),
    "three_quarters": lambda L: round(3 * L / 4),
}


def two_thirds_layer(n_layers: int) -> int:
    """The anchor layer l* used for the token trajectory."""
    return round(2 * n_layers / 3)


def resolve_layer(spec, n_layers: int) -> int:
    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise TypeError("layer spec must not be a bool")
    if isinstance(spec, int):
        idx = spec if spec >= 0 else n_layers + 1 + spec
        _check(idx, n_layers)
        return idx
    if isinstance(spec, float):
        if not 0.0 <= spec <= 1.0:
            raise ValueError(f"float layer spec must be in [0, 1]; got {spec}")
        return round(spec * n_layers)
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in _KEYWORDS:
            return _KEYWORDS[s](n_layers)
        if s.lstrip("-").isdigit():
            return resolve_layer(int(s), n_layers)
        try:
            return round(float(s) * n_layers) if 0.0 <= float(s) <= 1.0 else resolve_layer(int(float(s)), n_layers)
        except ValueError:
            pass
    raise ValueError(f"Cannot resolve layer spec {spec!r}")


def resolve_layers(spec, n_layers: int) -> list[int]:
    """Resolve a spec (or iterable / slice string) to a sorted unique list."""
    if isinstance(spec, str) and ":" in spec:
        parts = [int(p) if p else None for p in spec.split(":")]
        sl = slice(*parts)
        return sorted(set(range(n_layers + 1)[sl]))
    if isinstance(spec, (list, tuple, set)):
        out: set[int] = set()
        for item in spec:
            out.add(resolve_layer(item, n_layers))
        return sorted(out)
    return [resolve_layer(spec, n_layers)]


def all_layers(n_layers: int) -> list[int]:
    return list(range(n_layers + 1))


def _check(idx: int, n_layers: int) -> None:
    if not 0 <= idx <= n_layers:
        raise ValueError(f"layer index {idx} out of range [0, {n_layers}]")


def summarize(spec, extra: Iterable, n_layers: int) -> dict:
    star = resolve_layer(spec, n_layers)
    extras = resolve_layers(list(extra), n_layers) if extra else []
    return {"n_layers": n_layers, "star_layer": star, "extra_token_layers": extras}
