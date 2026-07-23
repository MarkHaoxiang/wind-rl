import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

# Import the full system entrypoint first: importing `mava.utils.make_env` (or our
# env module, which pulls it in) ahead of the systems package trips a circular
# import inside Mava (network_utils <-> make_env). Loading ff_mappo first lets
# Mava's own modules initialise in dependency order.
from mava.systems.ppo.anakin.ff_mappo import run_experiment  # isort: skip
import mava.utils.make_env as mava_make_env

from windrl_train.env import make_windfarm_envs

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
def main(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
