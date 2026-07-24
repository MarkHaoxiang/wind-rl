# Architecture

A map of the two packages and how `experiments/` uses them. Dependency
direction: `windrl-engine` is consumed by `windrl-train`. One uv workspace, one
py3.12 venv, fully declared in the package `pyproject.toml`s (`uv sync`).

## Packages

- `packages/windrl-engine`: pure-JAX wind-farm simulator; depends on floris only
  as a data/reference source, no torch/wfcrl dependency.
- `packages/windrl-train`: experiment harness over `windrl-engine` (config base,
  `WIND_RL_*` settings, verdict scoring, wandb logging); the RL trainer is being
  rewritten in-repo.
- `experiments/`: numbered frameworks (not single runs) that orchestrate the
  packages and assert pass/fail verdicts in code.

## `windrl-engine`

From-scratch JAX reimplementation of the WFCRL/FLORIS GCH wake model,
verified live to ~1e-13 against FLORIS 4.6.6 (a pinned runtime dependency;
WFCRL is not);
pure-functional (`NamedTuple` PyTrees, single-farm cores, batched via
`jit(vmap)`), layered `farm` -> `physics` -> `env` (plus `design` and
`analysis`), each layer importing only from lower ones.

- `farm/`: `turbine.py` (`TurbineSpec`, NREL-5MW read from FLORIS's packaged
  `nrel_5MW.yaml` at import via `farm/floris_tables.py`), `layout.py`
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
- `tests/`: live parity against FLORIS 4.6.6, physics invariants, and a
  beartype import hook that makes jaxtyping shape annotations
  runtime-checked.

## `windrl-train`

The experiment harness shared with `experiments/`; the RL trainer is being
rewritten in-repo.

- `config.py`: `Config`, the pydantic v2 `extra="forbid"` base every typed
  experiment config extends, with OmegaConf-backed YAML + dotlist loading.
- `settings.py`: `WindRlSettings`, the one `WIND_RL_*` env-var contract in the
  repo.
- `verdict.py`: `windowed_delta`, the per-run learning-signal score frameworks
  gate on.
- `logging.py`: `WandbLogger`, a wandb run wrapper honoring
  `WIND_RL_WDIR`/`WIND_RL_WANDB_MODE` via `WindRlSettings`.

## Outputs

Everything (checkpoints, logged layouts, wandb runs) defaults under the
gitignored `outputs/` at the repo root via `WindRlSettings`/`WIND_RL_WDIR`.
