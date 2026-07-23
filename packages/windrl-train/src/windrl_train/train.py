import json
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Import the full system entrypoint first: importing `mava.utils.make_env` (or our
# env module, which pulls it in) ahead of the systems package trips a circular
# import inside Mava (network_utils <-> make_env). Loading ff_mappo first lets
# Mava's own modules initialise in dependency order.
from mava.systems.ppo.anakin import ff_mappo  # isort: skip
from mava.systems.ppo.anakin.ff_mappo import run_experiment  # isort: skip
import mava.utils.make_env as mava_make_env
from mava.types import Metrics
from mava.utils.logger import LogEvent, MavaLogger

from windrl_train.env import make_windfarm_envs

#: When set, ``main`` dumps ``{"final_eval", "eval_series"}`` here after training so
#: an out-of-process orchestrator (e.g. an experiments framework in a different
#: venv) can read the deterministic-eval trajectory without parsing console logs.
_METRICS_PATH_ENV = "WINDRL_TRAIN_METRICS_PATH"

_last_logger: "_EvalRecordingLogger | None" = None


class _EvalRecordingLogger(MavaLogger):
    """MavaLogger that also retains the per-evaluation mean episode return."""

    def __init__(self, config: DictConfig, *args: object, **kwargs: object) -> None:
        super().__init__(config, *args, **kwargs)  # type: ignore[arg-type]
        self.eval_series: list[dict[str, float]] = []
        self.absolute_return: float | None = None
        global _last_logger
        _last_logger = self

    def log(self, metrics: Metrics, t: int, t_eval: int, event: LogEvent) -> None:
        if "episode_return" in metrics:
            if event is LogEvent.EVAL:
                self.eval_series.append(
                    {
                        "timestep": float(t),
                        "episode_return": float(np.mean(metrics["episode_return"])),
                    }
                )
            elif event is LogEvent.ABSOLUTE:
                self.absolute_return = float(np.mean(metrics["episode_return"]))
        super().log(metrics, t, t_eval, event)


# run_experiment reads `MavaLogger` from its own module globals at call time, so
# swapping the symbol records the eval series with zero edits to Mava source.
ff_mappo.MavaLogger = _EvalRecordingLogger  # type: ignore[misc]

_WINDFARM_ENV_NAME = "WindFarm"
_mava_make = mava_make_env.make


def _make(config: DictConfig, add_global_state: bool = False) -> tuple:
    if config.env.env_name == _WINDFARM_ENV_NAME:
        return make_windfarm_envs(config, add_global_state)
    return _mava_make(config, add_global_state)


# Mava's env factory is a fixed registry with no extension hook; routing our env
# through it here (rather than editing Mava) keeps the fork at zero source edits.
# ff_mappo.run_experiment looks up `environments.make` at call time, so patching
# the module attribute is sufficient.
mava_make_env.make = _make

# `pkg://mava.configs` resolves via a namespace package (loader=None), which makes
# Hydra warn "not available" even though the search path is used correctly.
warnings.filterwarnings("ignore", message=r"provider=hydra\.searchpath.*mava\.configs")


@hydra.main(config_path="configs", config_name="ff_mappo", version_base="1.2")
def main(config: DictConfig) -> float:
    OmegaConf.set_struct(config, False)
    eval_performance = run_experiment(config)
    metrics_path = os.environ.get(_METRICS_PATH_ENV)
    if metrics_path is not None and _last_logger is not None:
        Path(metrics_path).write_text(
            json.dumps(
                {
                    "final_eval": eval_performance,
                    "absolute_return": _last_logger.absolute_return,
                    "eval_series": _last_logger.eval_series,
                }
            )
        )
    return eval_performance


if __name__ == "__main__":
    main()
