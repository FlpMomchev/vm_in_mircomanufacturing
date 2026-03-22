"""vm_micro.utils.config
~~~~~~~~~~~~~~~~~~~~~~~
Unified YAML + CLI-override configuration loader.

Usage example
-------------
    cfg = load_config("configs/airborne.yaml")

    # CLI overrides (e.g. from argparse unknown args):
    cfg = apply_overrides(cfg, ["--target_sr=96000", "--dwt_wavelet=db4"])

    # Merge two config dicts (right wins):
    merged = merge_configs(base_cfg, override_cfg)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_configs(*paths: str | Path) -> dict[str, Any]:
    """Load and merge multiple YAML config files left-to-right (right wins)."""
    result: dict[str, Any] = {}
    for p in paths:
        result = merge_configs(result, load_config(p))
    return result


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge *override* into *base*.  Override always wins on scalar conflicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply a list of CLI-style ``--key=value`` or ``--section.key=value`` strings.

    Supported types: bool, int, float, str, null (None).

    Examples
    --------
    ::

        apply_overrides(cfg, ["--epochs=30", "--dl.lr=1e-4", "--use_amp=false"])
    """
    cfg = copy.deepcopy(cfg)
    for token in overrides:
        token = token.lstrip("-")
        if "=" not in token:
            # bare flag → True
            key_path, value_str = token, "true"
        else:
            key_path, value_str = token.split("=", 1)

        keys = key_path.split(".")
        parsed = _parse_scalar(value_str)

        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = parsed

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_scalar(s: str) -> Any:
    """Parse a CLI value string into the most specific Python type."""
    if s.lower() in {"true", "yes"}:
        return True
    if s.lower() in {"false", "no"}:
        return False
    if s.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Retrieve a value by dotted key path, returning *default* if absent."""
    keys = dotted_key.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
