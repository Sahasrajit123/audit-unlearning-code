"""YAML config loading with ``include`` support and dotted CLI overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

__all__ = [
    "load_config",
    "deep_merge",
    "apply_overrides",
    "config_hash",
    "config_source_files",
]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``; ``override`` wins on scalars."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    """Parse an override value as YAML so ints/floats/bools/lists work naturally."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Only existing leaves may be overridden.

    Refusing unknown keys turns a typo into an error instead of a silently ignored
    setting -- which in this project could mean an experiment running with a
    different configuration than the one you believe you launched.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value; got {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config section {'.'.join(parts[:-1])!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {dotted!r}")
        node[parts[-1]] = _coerce(raw)
    return out


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
    _seen: set | None = None,
) -> Dict[str, Any]:
    """Load a YAML config, resolving an optional ``include:`` list first.

    ``include`` paths are relative to the including file. Later includes override
    earlier ones; the including file overrides all of its includes.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular include involving {path}")
    _seen = _seen | {path}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    includes: List[str] = raw.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: Dict[str, Any] = {}
    for inc in includes:
        merged = deep_merge(merged, load_config(path.parent / inc, None, _seen))
    merged = deep_merge(merged, raw)

    if overrides:
        merged = apply_overrides(merged, overrides)
    return merged


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable hash of the effective config, saved with every result."""
    import hashlib

    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def config_source_files(path: str | Path, _seen: set | None = None) -> list:
    """Every YAML file that contributes to ``path``, including transitive includes.

    Returned in load order (deepest include first, the named file last), which is the
    order :func:`load_config` merges them. Used to snapshot the exact sources into a
    run directory, so a result can be reproduced from what was actually on disk rather
    than from whatever the config files say today.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        return []
    _seen = _seen | {path}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    includes = raw.get("include", []) or []
    if isinstance(includes, str):
        includes = [includes]

    out: list = []
    for inc in includes:
        for f_ in config_source_files(path.parent / inc, _seen):
            if f_ not in out:
                out.append(f_)
    out.append(path)
    return out
