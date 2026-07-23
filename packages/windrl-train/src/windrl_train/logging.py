import os
from pathlib import Path
from typing import Any

import jax
from mava.utils.logger import BaseLogger, LogEvent

_WANDB_MODE_ENV = "WIND_RL_WANDB_MODE"
_WDIR_ENV = "WIND_RL_WDIR"
_DEFAULT_MODE = "online"
_DEFAULT_WDIR = "outputs"


def _resolved_mode() -> str:
    mode = os.environ.get(_WANDB_MODE_ENV, _DEFAULT_MODE)
    if mode not in ("online", "offline", "disabled"):
        raise ValueError(
            f"{_WANDB_MODE_ENV} must be online|offline|disabled, got {mode!r}"
        )
    return mode


def _resolved_dir() -> Path:
    return Path(os.environ.get(_WDIR_ENV, _DEFAULT_WDIR)).expanduser().resolve()


class WandbLogger(BaseLogger):
    """Weights & Biases backend for Mava's ``MultiLogger``.

    Mode and output directory follow the ``WIND_RL_WANDB_MODE`` /
    ``WIND_RL_WDIR`` env-var contract (mirrored, not imported). The run ``name``
    must not encode the seed — seeds share a name so wandb's group-by-name shows
    the seed distribution; the seed lives in ``tags``/``config`` instead.
    """

    def __init__(
        self,
        base_exp_path: os.PathLike,
        unique_token: str,
        system_name: str,
        *,
        project: str,
        name: str,
        seed: int,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import wandb

        wdir = _resolved_dir()
        wdir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=[*(tags or []), f"seed_{seed}"],
            mode=_resolved_mode(),
            dir=str(wdir),
            reinit=True,
        )
        # A single ``timestep`` x-axis for every event so eval return plots over
        # environment steps and cross-event logs never fight over wandb's step.
        self._run.define_metric("timestep")
        self._run.define_metric("*", step_metric="timestep")

    def log_stat(
        self, key: str, value: float, step: int, eval_step: int, event: LogEvent
    ) -> None:
        if isinstance(value, jax.Array):
            value = value.item()
        self._run.log({f"{event.value}/{key}": value, "timestep": step})

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def stop(self) -> None:
        self._run.finish()
