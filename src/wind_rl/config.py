"""Pydantic configuration base with OmegaConf-backed loading (the "pydra" pattern).

Configs are strict pydantic v2 models (unknown fields raise) that can be built
directly, loaded from a YAML file, or constructed from an already-parsed
:class:`~omegaconf.DictConfig`. File loading additionally supports Hydra-style
dotlist overrides (e.g. ``["n_turbines=8", "max_steps=200"]``) merged on top of
the file before validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base class for all wind-rl configuration objects."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_raw(cls, cfg: DictConfig) -> Self:
        """Validate an already-built :class:`DictConfig` into this config type."""
        container: Any = OmegaConf.to_container(cfg, resolve=True)
        return cls.model_validate(container)

    @classmethod
    def from_file(cls, path: Path, overrides: Sequence[str] | None = None) -> Self:
        """Load this config from a YAML file, optionally patched with dotlist overrides."""
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected a mapping at {path}, got {type(cfg).__name__}")
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            if not isinstance(cfg, DictConfig):  # pragma: no cover - defensive
                raise TypeError("Merging overrides did not produce a mapping")
        return cls.from_raw(cfg)
