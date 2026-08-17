"""Recursive OmegaConf loader for the additive Event80 method configs."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_event80_config(path: str | Path) -> DictConfig:
    config_path = Path(path).expanduser().resolve()
    current = OmegaConf.load(config_path)
    base = current.get("extends") or current.get("base_config")
    if not base:
        return current
    base_path = Path(str(base))
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    current = OmegaConf.create(OmegaConf.to_container(current, resolve=False))
    current.pop("extends", None)
    current.pop("base_config", None)
    return OmegaConf.merge(load_event80_config(base_path), current)
