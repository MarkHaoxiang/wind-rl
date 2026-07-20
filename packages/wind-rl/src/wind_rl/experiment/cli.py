"""Shared Hydra glue for experiment entry points."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from hydra import compose, initialize_config_dir

from wind_rl.config import Config

C = TypeVar("C", bound=Config)


def compose_experiment(
    conf_dir: Path, config_cls: type[C], overrides: Sequence[str]
) -> C:
    """Compose ``conf_dir/config.yaml`` (patched with ``overrides``) into ``config_cls``."""
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=list(overrides))
    return config_cls.from_raw(cfg)
