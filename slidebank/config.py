from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict


def _expand_config_values(value: Any) -> Any:
    """Expand explicitly supplied environment references without executing code."""
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        missing = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", expanded)
        if missing:
            names = sorted({left or right for left, right in missing})
            raise ValueError(f"Unset environment variable(s) in config: {', '.join(names)}")
        return expanded
    return value


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {cfg_path}")
    return _expand_config_values(data)


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def get_nested(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
