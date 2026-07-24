# Architecture

A map of the two packages and how `experiments/` uses them. Dependency
direction: `windrl-engine` is consumed by `windrl-train`. One uv workspace,
one py3.12 venv (Mava pins jax==0.5.3 / py<3.13); Mava is installed editable
from a clone, not declared as a dependency (see `packages/windrl-train/README.md`).

## Packages

- `packages/windrl-engine`: pure-JAX wind-farm simulator, no torch/wfcrl/FLORIS
  dependency.
- `packages/windrl-train`: Mava MAPPO trainer over `windrl-engine`, plus the
  experiment harness (config base, `WIND_RL_*` settings, verdict scoring).
- `experiments/`: numbered frameworks (not single runs) that orchestrate the
  packages and assert pass/fail verdicts in code, e.g. `0002_mappo_baseline`
  runs each seed in a fresh `windrl_train.train` subprocess (Hydra/global-state
  isolation per run).

## `windrl-engine`

From-scratch JAX reimplementation of the WFCRL/FLORIS GCH wake model,
verified to ~1e-13 against a frozen FLORIS 4.6.6 golden
(`tests/goldens/floris_v4.6.6.npz`), so neither WFCRL nor FLORIS is a
dependency;
pure-functional (`NamedTuple` PyTrees, single-farm cores, batched via
`jit(vmap)`), layered `farm` -> `physics` -> `env` (plus `design` and
`analysis`), each layer importing only from lower ones.

- `farm/`: `turbine.py` (`TurbineSpec`, NREL-5MW loaded from generated
  package data `farm/data/nrel5mw_v4.npz`), `layout.py`
  (`FarmLayout` + registry `turb3_row1`/`ablaincourt`/`horns_rev2`),
  `wind.py` (`WindCondition`, sampling, `WindRose`), `state.py` (`FarmState`:
  yaws + wind).
- `physics/`: one module per GCH stage (`frame`, `flow`, `thrust`,
  `deflection`, `transverse`, `deficit`, `turbulence`, `power`) composed by
  `solver.py`'s sequential upstream-to-downstream solve, with a
  `fidelity="floris"|"corrected"` flag choosing whether to reproduce or fix
  the reference's numerical quirks.
- `env/`: WFCRL-compatible MARL env (delta-yaw actions, duty-cycle limiter,
  reward) — `spaces.py`, `actions.py`, `config.py` (`WindFarmEnvConfig`),
  `env.py` (`BatchedWindFarmEnv` with per-env layouts and auto-reset; reward is
  a dependency-injected `RewardFn`, defaulting to the `wfcrl_reward` factory).
- `design/`: `Designer` type alias (a pure `(key, batch) -> layouts` callable,
  no `Protocol` — current designers are stateless closures), `SiteSpec`,
  `project_feasible` min-spacing projection, random/grid designers.
- `analysis/`: `flow_viz.py`, `metrics.py` (wind-rose power), `plots.py`.
- `tests/`: golden parity against FLORIS 4.6.6, physics invariants, and a
  beartype import hook that makes jaxtyping shape annotations
  runtime-checked.

## `windrl-train`

Zero Mava source edits; Mava is pinned to a git SHA and installed editable
from a clone (see the package README).

- `env.py`: Jumanji `MarlEnv` wrapper — the turbine axis is the agent axis;
  Mava vmaps `num_envs` itself, so the wrapper is unbatched.
- `train.py`: monkeypatches the env factory into Mava's `ff_mappo`
  entrypoint and adds eval-series JSON export.
- `networks.py`: permutation-equivariant GCN torsos (`network=gcn`) beside
  Mava's default per-agent MLP (`network=mlp`).
- `logging.py`: wandb backend honoring `WIND_RL_WDIR`/`WIND_RL_WANDB_MODE` via
  `WindRlSettings`.
- `configs/`: Hydra composition over Mava's own config groups.
- `settings.py`: `WindRlSettings`, the one `WIND_RL_*` env-var contract in the
  repo.
- `config.py`: `Config`, the pydantic v2 `extra="forbid"` base every typed
  experiment config extends.
- `verdict.py`: `windowed_delta`, the per-run learning-signal score frameworks
  gate on.

## Outputs

Everything (checkpoints, logged layouts, wandb runs) defaults under the
gitignored `outputs/` at the repo root via `WindRlSettings`/`WIND_RL_WDIR`.
