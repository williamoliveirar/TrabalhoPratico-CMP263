"""Configuration loading: configs/<name>.yaml, with machine-specific paths overridden by configs/local.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(name: str = "base") -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
    local = ROOT / "configs" / "local.yaml"
    if name == "base" and local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Absolute path for cfg["paths"][key]; relative entries are resolved against the repository root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
