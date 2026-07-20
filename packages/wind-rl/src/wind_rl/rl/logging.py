"""wandb integration and pure PPO logging helpers for the MAPPO trainer.

:func:`explained_variance` and :func:`clip_fraction` are pure and unit-tested;
:class:`RunLogger` isolates the optional wandb dependency so the trainer stays
runnable -- and testable -- with wandb disabled.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from wandb.sdk.wandb_run import Run

    from wind_rl.experiment.settings import WindRlSettings
    from wind_rl.rl.trainer import TrainingConfig

#: wandb x-axis every metric is plotted against (cumulative env frames).
X_AXIS_METRIC = "train/total_frames"


def explained_variance(value_target: torch.Tensor, value_pred: torch.Tensor) -> float:
    """1 - Var(target - pred) / Var(target); <=1, negative when worse than the mean."""
    var_target = float(value_target.var(unbiased=False))
    if var_target <= 0.0:
        return 0.0
    residual_var = float((value_target - value_pred).var(unbiased=False))
    return 1.0 - residual_var / var_target


def clip_fraction(log_ratio: torch.Tensor, clip_epsilon: float) -> float:
    """Fraction of samples whose likelihood ratio leaves ``[1-eps, 1+eps]``.

    Matches torchrl's PPO clip-fraction: the comparison is on the log-ratio
    against ``[log(1-eps), log(1+eps)]``.
    """
    low, high = math.log1p(-clip_epsilon), math.log1p(clip_epsilon)
    clipped = log_ratio.clamp(low, high)
    return float((clipped != log_ratio).to(torch.float32).mean())


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"
    return out.stdout.strip()


class RunLogger:
    """Optional wandb run: a no-op when wandb is disabled or unrequested."""

    def __init__(self, cfg: TrainingConfig, settings: WindRlSettings) -> None:
        self._run: Run | None = None
        if not cfg.logging.use_wandb or settings.wandb_mode == "disabled":
            return
        import wandb

        config = cfg.model_dump()
        config["git_commit"] = git_commit()
        self._run = wandb.init(
            project=cfg.logging.project,
            name=cfg.experiment_name,
            mode=settings.wandb_mode,
            config=config,
        )
        self._run.define_metric(X_AXIS_METRIC)
        self._run.define_metric("*", step_metric=X_AXIS_METRIC)

    @property
    def url(self) -> str | None:
        return None if self._run is None else self._run.url

    def log(
        self,
        metrics: dict[str, float],
        images: dict[str, NDArray[np.uint8]] | None = None,
    ) -> None:
        if self._run is None:
            return
        import wandb

        payload: dict[str, object] = dict(metrics)
        for key, image in (images or {}).items():
            payload[key] = wandb.Image(image)
        self._run.log(payload)

    def log_artifact(self, path: Path, name: str, artifact_type: str = "model") -> None:
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
