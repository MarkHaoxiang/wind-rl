# Architecture

A map of the three packages and how `experiments/` uses them. Dependency
direction: `windrl-engine` is consumed by both `windrl-train` and `wind-rl`;
neither of those depends on the other.

## Packages

- `packages/windrl-engine`: pure-JAX wind-farm simulator, in the root uv
  workspace, no torch/wfcrl/FLORIS dependency.
- `packages/windrl-train`: Mava MAPPO trainer over `windrl-engine`, excluded
  from the root workspace with its own Python 3.12 / jax 0.5.3 venv (Mava
  pins conflict with the root py3.13 / jax 0.11 stack).
- `packages/wind-rl`: experiment harness (sweep/table/verdict machinery,
  config base, scenario definitions), in the root workspace, torch-free.
- `experiments/`: numbered frameworks (not single runs) that orchestrate
  across the packages and assert pass/fail verdicts in code, e.g.
  `0002_mappo_baseline` shells `subprocess` calls into the `windrl-train`
  venv per seed.

## `windrl-engine`

From-scratch JAX reimplementation of the WFCRL/FLORIS GCH wake model,
verified to ~1e-13 against frozen FLORIS goldens
(`tests/goldens/floris_v3.5.npz`, `floris_v4.6.6.npz`,
`wfcrl_env_trajectories.npz`), so neither WFCRL nor FLORIS is a dependency;
pure-functional (`NamedTuple` PyTrees, single-farm cores, batched via
`jit(vmap)`), layered `farm` -> `physics` -> `env` (plus `design` and
`analysis`), each layer importing only from lower ones.

- `farm/`: `turbine.py` (`TurbineSpec`, NREL-5MW `v3`/`v4`), `layout.py`
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
  `env.py` (`BatchedWindFarmEnv` with per-env layouts and auto-reset).
- `design/`: `Designer` type alias (a pure `(key, batch) -> layouts` callable,
  no `Protocol` — current designers are stateless closures), `SiteSpec`,
  `project_feasible` min-spacing projection, random/grid designers.
- `analysis/`: `flow_viz.py`, `metrics.py` (wind-rose power), `plots.py`.
- `tests/`: golden parity against FLORIS 3.5/4.6.6 and recorded WFCRL
  trajectories, physics invariants, and a beartype import hook that makes
  jaxtyping shape annotations runtime-checked.

## `windrl-train`

Zero Mava source edits; Mava is pinned to a git SHA and installed editable
from a clone (see the package README).

- `env.py`: Jumanji `MarlEnv` wrapper — the turbine axis is the agent axis;
  Mava vmaps `num_envs` itself, so the wrapper is unbatched.
- `train.py`: monkeypatches the env factory into Mava's `ff_mappo`
  entrypoint and adds eval-series JSON export.
- `networks.py`: permutation-equivariant GCN torsos (`network=gcn`) beside
  Mava's default per-agent MLP (`network=mlp`).
- `logging.py`: wandb backend honoring `WIND_RL_WDIR`/`WIND_RL_WANDB_MODE`
  (mirrored, not imported — this venv has no `wind_rl`).
- `configs/`: Hydra composition over Mava's own config groups.

## `wind-rl`

- `experiment/`: `settings.py` (`WindRlSettings`, the `WIND_RL_*` env-var
  contract), `sweep.py`/`table.py`/`verdict.py` (per-run results ->
  aggregated table -> pass/fail gates), `cli.py` (shared Hydra glue).
- `config.py`: `Config`, the pydantic v2 `extra="forbid"` base every typed
  config extends.
- `scenario.py`: `ScenarioConfig` — map geometry and control parameters
  shared across layout-feasibility work.
- `utils/`: `thread_pinning.py` only.

## Outputs

Everything (checkpoints, logged layouts, wandb runs) defaults under the
gitignored `outputs/` at the repo root via `WindRlSettings`/`WIND_RL_WDIR`.
