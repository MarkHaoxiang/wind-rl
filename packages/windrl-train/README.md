# windrl-train

Mava MAPPO training on the `windrl-engine` JAX wind farm.

The wrapper (`windrl_train.env.WindFarm`) presents the engine's single-farm
functional `reset`/`step` core as a Mava/Jumanji `MarlEnv`: the turbine axis is
the agent axis, each agent has a continuous delta-yaw action in `[-1, 1]`
(rescaled to the engine's `[-yaw_step, +yaw_step]` box), and the shared scalar
reward is repeated per agent. Mava does its own `jax.vmap` over `num_envs`, so
the wrapper is unbatched. `train.py` routes the env into Mava's `ff_mappo`
entrypoint (continuous head auto-selected from the `BoundedArray` action spec)
with zero edits to Mava source.

## Why this package is version-isolated

Mava is GitHub-only (never released to PyPI as the JAX system) and pins
`jax==0.5.3`, `jaxlib==0.5.3`, `numpy==1.26.4`, `python>=3.10,<3.13`. The root
workspace runs Python 3.13 / jax 0.11, and `windrl-engine` targets `jax>=0.5.3`.
No single resolution satisfies both jax pins or both Python bounds, so this
package is **excluded from the root `[tool.uv.workspace]`** and gets its own
Python 3.12 / jax 0.5.3 venv. Mava is pinned to a git SHA
(`e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf`).

## Setup

From the repo root:

```bash
# 1. Create the isolated venv (Python 3.12).
uv venv --python 3.12 packages/windrl-train/.venv
VENV=$(pwd)/packages/windrl-train/.venv

# 2. Install Mava editable from a clone at the pinned SHA. (A PEP 508
#    "mava @ git+..." dependency does not work: the distro name is id-mava,
#    and a non-editable git wheel drops mava/configs, breaking Hydra's
#    pkg://mava.configs searchpath. Editable-from-clone is Mava's own
#    documented install path.)
git clone https://github.com/instadeepai/Mava.git /tmp/mava-checkout
git -C /tmp/mava-checkout checkout e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python -e /tmp/mava-checkout

# 3. Install the engine and this package.
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python -e packages/windrl-engine
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python --no-deps -e packages/windrl-train
```

## Smoke run

```bash
JAX_PLATFORMS=cpu packages/windrl-train/.venv/bin/python -m windrl_train.train \
  env.kwargs.layout=turb3_row1 env.kwargs.horizon=100 \
  arch.num_envs=8 system.update_batch_size=1 system.rollout_length=32 \
  system.num_updates=16 arch.num_evaluation=4 arch.num_eval_episodes=2 \
  system.num_minibatches=2 arch.absolute_metric=False
```

Common Hydra overrides: `env.kwargs.layout` (`turb3_row1` | `ablaincourt` |
`horns_rev2`), `env.kwargs.horizon`, `env.kwargs.yaw_step`,
`env.kwargs.load_coef`, `arch.num_envs`, `system.total_timesteps`. Logging is
Mava's own console logger; enable others via `logger.loggers.<name>.enabled`.

## Networks

`network=mlp` (Mava's per-agent MLP) is the default. `network=gcn` selects the
permutation-equivariant GCN torsos in `windrl_train.networks`: the actor builds
a dense row-normalized Gaussian adjacency from the turbine `(x, y)` (the last
two `agents_view` channels, added by the env wrapper) and runs 2 residual
message-passing rounds shared across turbines; the centralised critic runs the
same GCN over the global state and mean-pools to a permutation-invariant value.
Append `network=gcn` to any run, e.g. the smoke command above.

The equivariance property (permuting turbines permutes actor means identically,
leaves the critic value unchanged) is checked in `tests/test_equivariance.py`:

```bash
# standalone
JAX_PLATFORMS=cpu packages/windrl-train/.venv/bin/python \
  packages/windrl-train/tests/test_equivariance.py
# or under pytest
cd packages/windrl-train && JAX_PLATFORMS=cpu ./.venv/bin/python -m pytest tests
```

## Checks

`ruff`/`ruff format` run from the root venv
(`uv run ruff check packages/windrl-train/src`). Type-checking runs inside this
venv (`cd packages/windrl-train && ./.venv/bin/mypy src`), because Mava and
jumanji live only here, not in the root py3.13 venv.
