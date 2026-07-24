# windrl-train

Mava MAPPO training on the `windrl-engine` JAX wind farm, plus the experiment
harness (`config`/`settings`/`verdict`) shared with `experiments/`.

The wrapper (`windrl_train.env.WindFarm`) presents the engine's single-farm
functional `reset`/`step` core as a Mava/Jumanji `MarlEnv`: the turbine axis is
the agent axis, each agent has a continuous delta-yaw action in `[-1, 1]`
(rescaled to the engine's `[-yaw_step, +yaw_step]` box), and the shared scalar
reward is repeated per agent. Mava does its own `jax.vmap` over `num_envs`, so
the wrapper is unbatched. `train.py` routes the env into Mava's `ff_mappo`
entrypoint (continuous head auto-selected from the `BoundedArray` action spec)
with zero edits to Mava source.

## Why one py3.12 venv, and why Mava is installed by hand

Mava is GitHub-only (never released to PyPI as the JAX system) and pins
`jax==0.5.3`, `jaxlib==0.5.3`, `numpy==1.26.4`, `python>=3.10,<3.13`. The whole
repo targets this stack: the root workspace is pinned to Python 3.12, and both
members (`windrl-engine`, `windrl-train`) co-resolve `jax==0.5.3`.

Mava is **not declared** as a dependency and is installed **editable from a
clone** at a pinned git SHA (`e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf`):

- A PEP 508 `mava @ git+...` spec hard-fails — Mava's distribution name is
  `id-mava`, and uv enforces the declared name against the build metadata.
- A non-editable git wheel drops `mava/configs/` (no `__init__.py`, no
  package-data include), so Hydra's `pkg://mava.configs` searchpath can't find
  e.g. `network/mlp`. Editable keeps `mava.__path__` on the checkout, so the
  configs are found on disk. This is Mava's own upstream-recommended install.

Because Mava isn't declared, a plain `uv sync` **prunes its whole closure**.
After the initial install, use `uv sync --inexact` for dependency changes (it
preserves Mava), and run every check/experiment with `uv run --no-sync` so the
frozen Mava install (its own scipy/jax pins) is never reconciled away.

## Setup

From the repo root:

```bash
# 1. Sync the workspace (py3.12: windrl-engine + windrl-train + declared deps).
uv sync

# 2. Install Mava editable from a clone at the pinned SHA (brings its ~150-pkg
#    closure: jax/jaxlib, flax, jumanji, jaxmarl, ...).
git clone https://github.com/instadeepai/Mava.git /tmp/mava-checkout
git -C /tmp/mava-checkout checkout e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf
uv pip install -e /tmp/mava-checkout
```

If a later `uv sync` prunes Mava, re-run step 2 (or prefer `uv sync --inexact`).

## Smoke run

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m windrl_train.train \
  env.kwargs.layout=turb3_row1 env.kwargs.horizon=100 \
  arch.num_envs=8 system.update_batch_size=1 system.rollout_length=32 \
  system.num_updates=16 arch.num_evaluation=4 arch.num_eval_episodes=2 \
  system.num_minibatches=2 arch.absolute_metric=False
```

Common Hydra overrides: `env.kwargs.layout` (`turb3_row1` | `ablaincourt` |
`horns_rev2`), `env.kwargs.horizon`, `env.kwargs.yaw_step`,
`env.kwargs.load_coef`, `arch.num_envs`, `system.total_timesteps`. Logging is
Mava's own console logger; enable others via `logger.loggers.<name>.enabled`.

## GPU (NVIDIA CUDA 12)

GPU is **opt-in** and never a hard dependency: CI is CPU-only and installs this
package `--no-deps`, so it must never pull the multi-GB CUDA wheels. The venv is
otherwise identical to the CPU setup above — the CUDA support is the `jax[cuda12]`
plugin layered on top of Mava's CPU jaxlib.

Install order is load-bearing. Mava re-pins jax during its editable install, so
the CUDA plugin (declared as this package's `gpu` extra, `jax[cuda12]==0.5.3`)
goes on **last**, or it gets clobbered:

```bash
VENV=$(pwd)/packages/windrl-train/.venv
# 1-3. as in Setup above (Mava editable, then windrl-engine, then this package).
# 4. CUDA plugin LAST — must match Mava's jax==0.5.3 pin.
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python "jax[cuda12]==0.5.3"
```

The `jax[cuda12]==0.5.3` extra bundles CUDA 12.9 wheels, which cover Blackwell
(`sm_120`, e.g. RTX 5090) via a recent driver's forward-compatible PTX JIT — a
current NVIDIA driver (CUDA 12.8+ / 13.x runtime) is the only host requirement;
no system CUDA toolkit is needed.

Verify the plugin sees the GPU:

```bash
packages/windrl-train/.venv/bin/python -c "import jax; print(jax.devices())"
# -> [CudaDevice(id=0)]
```

**`JAX_PLATFORMS` semantics.** With the CUDA plugin installed, jax defaults to
the GPU. Force a backend explicitly:

- `JAX_PLATFORMS=cuda` (or unset) — run on the GPU.
- `JAX_PLATFORMS=cpu` — force CPU even with the plugin present (used by the
  smoke/equivariance commands above and by CI, so they stay deterministic and
  GPU-independent).

The engine's FLORIS golden parity tests pass on GPU at `float64` to the same
`rtol=1e-9` gate as CPU, so GPU is safe for evaluation as well as rollouts.
Practical guidance from `bench/` (RTX 5090): the GPU wins big for `horns_rev2`
(≈37× training throughput vs CPU) and for any large-batch solve, but for the
tiny `turb3_row1`/`ablaincourt` farms at small batch the CPU is competitive or
faster (GPU dispatch overhead dominates). `float64` throughput on consumer
Blackwell is ~10× slower than `float32` for large farms — keep rollouts in
`float32` and reserve `float64` for final evaluation.

## Weights & Biases

Mava ships console/neptune/tensorboard/json loggers but no wandb one, so
`windrl_train.logging.WandbLogger` (a `mava.utils.logger.BaseLogger` subclass,
same zero-edit injection style as `train._EvalRecordingLogger`) is registered in
`configs/ff_mappo.yaml` under `logger.loggers.wandb`, disabled by default.
Enable it and the eval-return-over-steps charts follow:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m windrl_train.train \
  env.kwargs.layout=turb3_row1 network=mlp system.num_updates=8000 \
  logger.loggers.wandb.enabled=true logger.loggers.wandb.project=windrl-train
```

Mode and output directory follow the `WIND_RL_*` env-var contract via
`WindRlSettings` (`windrl_train.settings`) — one definition of the contract in
the repo:

- `WIND_RL_WANDB_MODE` — `online` (default) | `offline` | `disabled`.
- `WIND_RL_WDIR` — output dir for wandb runs, Hydra run artifacts, and Mava's
  `base_exp_path` (default `outputs/` relative to the launch cwd — the
  repo-root `outputs/` is gitignored).

The run `name` (default `mappo-mlp-<layout>`) deliberately omits the seed so
seeds share a name and wandb group-by-name shows the seed distribution; the seed
goes in `tags` (`seed_<n>`) instead. wandb is pinned `>=0.19,<0.20` with an
explicit `protobuf>=3.20,<4`: Mava imports `tensorboard_logger` unconditionally
(whose bundled pb2 stubs break on protobuf>=4), so the trainer needs protobuf<4
to import at all; wandb>=0.20 ships pb2 stubs that require protobuf>=5 at
runtime, so 0.19 is the newest that co-resolves.

## Networks

`network=mlp` (Mava's per-agent MLP) is the default. `network=gcn` selects the
permutation-equivariant GCN torsos in `windrl_train.networks`: the actor builds
a dense row-normalized Gaussian adjacency from the turbine `(x, y)` (the last
two `agents_view` channels, added by the env wrapper) and runs 2 residual
message-passing rounds shared across turbines; the centralised critic runs the
same GCN over the global state and mean-pools to a permutation-invariant value.
Append `network=gcn` to any run, e.g. the smoke command above.

The equivariance property (permuting turbines permutes actor means identically,
leaves the critic value unchanged) is checked in `tests/test_equivariance.py`,
which runs in the shared suite (`uv run --no-sync pytest -q`).

## Checks

All four run from the repo root with `uv run --no-sync` (see the repo
`CLAUDE.md` Checks section). config/settings/verdict are strict-typed; the
Mava/jumanji-touching modules (`env`/`networks`/`train`/`logging`) get
per-module mypy relaxations at the workspace root.
