from typing import Any

import jax

from windrl_train.settings import WindRlSettings


class WandbLogger:
    """Weights & Biases run wrapper honoring the WIND_RL_* settings contract.

    The run ``name`` must not encode the seed — seeds share a name so wandb's
    group-by-name shows the seed distribution; the seed lives in ``tags``.
    """

    def __init__(
        self,
        *,
        project: str,
        name: str,
        seed: int,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import wandb

        settings = WindRlSettings()
        wdir = settings.resolved_wdir
        wdir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=[*(tags or []), f"seed_{seed}"],
            mode=settings.wandb_mode,
            dir=str(wdir),
            reinit=True,
        )
        # One shared ``timestep`` x-axis so eval-return plots track environment
        # steps and cross-event logs never fight over wandb's implicit step.
        self._run.define_metric("timestep")
        self._run.define_metric("*", step_metric="timestep")

    def log_stat(self, key: str, value: float, step: int, event: str) -> None:
        if isinstance(value, jax.Array):
            value = value.item()
        self._run.log({f"{event}/{key}": value, "timestep": step})

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def stop(self) -> None:
        self._run.finish()
