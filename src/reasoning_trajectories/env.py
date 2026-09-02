"""Optional ``.env`` loading.

If a ``.env`` file exists at (or above) the current working directory or the
package location, its values are loaded into ``os.environ`` **without**
overriding variables already set in the real environment.

``python-dotenv`` is used when installed; otherwise a tiny built-in parser
handles ``KEY=value`` / ``export KEY=value`` lines. Either way this is a no-op
when there is no ``.env`` file, so it is safe to call unconditionally.

Recognised variables (see ``.env``):
    NDIF_API_KEY / NNSIGHT_API_KEY   NDIF key for remote nnsight execution
    HF_TOKEN                         Hugging Face token for gated models (Llama, ...)
"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env(path: str | os.PathLike | None = None, *, override: bool = False) -> Path | None:
    """Load a ``.env`` file. Returns the path used, or ``None`` if none found.

    Called once automatically on package import; safe to call again (cheap
    no-op unless an explicit ``path`` is given).
    """
    global _LOADED
    if _LOADED and path is None:
        return None

    found = _resolve(path)
    _LOADED = True
    if found is None:
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(found, override=override)
    except ImportError:
        _load_fallback(found, override)
    return found


def _resolve(explicit) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (base, *base.parents):
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


def _load_fallback(path: Path, override: bool) -> None:
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val
