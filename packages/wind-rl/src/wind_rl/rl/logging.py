"""wandb integration and pure PPO logging helpers for the MAPPO trainer.

:func:`explained_variance` is pure and unit-tested; :class:`RunLogger`
isolates the optional wandb dependency so the trainer stays runnable -- and
testable -- with wandb disabled.
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


def param_norms(module: torch.nn.Module) -> dict[str, float]:
    """L2 norm per leaf submodule (keyed by dotted path) plus ``"total"``.

    Reads ``.detach()``ed parameters only -- no forward pass -- so it is cheap
    enough to call every logging step.
    """
    norms: dict[str, float] = {}
    total_sq = 0.0
    for name, leaf in module.named_modules():
        if next(leaf.children(), None) is not None:
            continue  # not a leaf; its parameters are counted by its children
        # .abs() (not .float()) so complex-valued params (e.g. SO2) are handled
        # correctly -- .float() on a complex tensor silently drops the imaginary part.
        leaf_sq = sum(float(p.detach().abs().pow(2).sum()) for p in leaf.parameters())
        if leaf_sq == 0.0:
            continue
        norms[name or "root"] = math.sqrt(leaf_sq)
        total_sq += leaf_sq
    norms["total"] = math.sqrt(total_sq)
    return norms


def checkpoint_aliases(
    iteration: int, upload_interval: int | None, *, is_final: bool
) -> list[str]:
    """wandb aliases for the checkpoint saved at ``iteration``; empty means skip upload."""
    aliases: list[str] = []
    if upload_interval is not None and iteration % upload_interval == 0:
        aliases.append(f"iter-{iteration}")
    if is_final:
        aliases.append("final")
    return aliases


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
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return None if self._run is None else self._run.url

    def log(
        self,
        metrics: dict[str, float],
        images: dict[str, NDArray[np.uint8]] | None = None,
        html: dict[str, str] | None = None,
    ) -> None:
        if self._run is None:
            return
        import wandb

        payload: dict[str, object] = dict(metrics)
        for key, image in (images or {}).items():
            payload[key] = wandb.Image(image)
        for key, document in (html or {}).items():
            payload[key] = wandb.Html(document, inject=False)
        self._run.log(payload)

    def log_artifact(
        self,
        path: Path,
        name: str,
        aliases: list[str] | None = None,
        artifact_type: str = "model",
    ) -> None:
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
