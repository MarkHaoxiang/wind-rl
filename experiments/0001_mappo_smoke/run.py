"""0001_mappo_smoke: MAPPO walking-skeleton smoke test on a 3-turbine FLORIS row.

Verdict (asserted in code): deterministic eval mean episode reward -- total farm
power under wake steering -- rises over training. Each evaluation resets with
freshly sampled wind (direction ~N(270, 20), Weibull speed), so a single eval
point is noisy; the verdict therefore compares the mean of the last third of
evaluations against the first third. PASS iff the last window strictly exceeds
the first. Prints PASS/FAIL and exits nonzero on FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from wind_rl.rl.trainer import MappoTrainer, TrainingConfig

_METRIC = "eval_episode_reward"


def _compose(overrides: list[str]) -> DictConfig:
    conf_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        return compose(config_name="config", overrides=overrides)


def _verdict(history: list[dict[str, float]]) -> tuple[bool, float, float]:
    evals = [m[_METRIC] for m in history if _METRIC in m]
    window = max(1, len(evals) // 3)
    first = sum(evals[:window]) / window
    last = sum(evals[-window:]) / window
    return last > first, first, last


def main() -> int:
    cfg = TrainingConfig.from_raw(_compose(sys.argv[1:]))
    history = MappoTrainer(cfg).run()

    evals = [m[_METRIC] for m in history if _METRIC in m]
    print("\neval trajectory:", " ".join(f"{v:.2f}" for v in evals))
    print(f"first / last iteration {_METRIC}: {evals[0]:.4f} / {evals[-1]:.4f}")

    passed, first, last = _verdict(history)
    print(f"first-window mean {_METRIC}: {first:.4f}")
    print(f"last-window  mean {_METRIC}: {last:.4f}")
    print(f"delta: {last - first:+.4f}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
