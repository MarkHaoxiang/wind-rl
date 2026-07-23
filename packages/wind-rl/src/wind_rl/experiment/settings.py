"""Process-wide settings, sourced from ``WIND_RL_*`` environment variables.

Replaces DiCoDe's hardcoded ``~/.diffusion_co_design`` working directory with a
configurable, env-overridable setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class WindRlSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WIND_RL_", extra="forbid")

    # Repo-root `outputs/` is gitignored, so default runs never dirty the tree.
    wdir: Path = Path("outputs")
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @property
    def resolved_wdir(self) -> Path:
        """``wdir`` with ``~`` expanded and resolved to an absolute path."""
        return self.wdir.expanduser().resolve()
