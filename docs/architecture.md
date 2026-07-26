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
`jit(vmap)`), layered `farm` -> `physics` -> `env` (plus `design`, `metrics`
and `viz`), each layer importing only from lower ones.

- `farm/`: `turbine.py` (`TurbineSpec`, NREL-5MW read from FLORIS's packaged
  `nrel_5MW.yaml` at import via `farm/floris_tables.py`), `layout.py`
  (`FarmLayout` + registry `turb3_row1`/`ablaincourt`/`horns_rev2`),
  `wind.py` (`WindCondition`, sampling, `WindRose`), `state.py` (`FarmState`:
  yaws + wind).
- `physics/`: one module per GCH stage (`frame`, `flow`, `thrust`,
  `deflection`, `transverse`, `deficit`, `turbulence`, `power`) composed by
  `solver.py`'s sequential upstream-to-downstream solve, with a
  `fidelity="floris"|"corrected"` flag choosing whether to reproduce or fix
  the reference's numerical quirks; `query_field.py` re-casts a converged
  solution onto arbitrary query points (the full-flow field pass).
- `env/`: WFCRL-compatible MARL env (delta-yaw actions, duty-cycle limiter,
  reward) — `spaces.py`, `actions.py`, `config.py` (`WindFarmEnvConfig`),
  `env.py` (`BatchedWindFarmEnv` with per-env layouts and auto-reset, exposing a
  pure `lax.scan`-safe `EnvState`/`reset_fn`/`step_fn` API that the stateful
  `reset`/`step`/`rollout` wrap; reward is a dependency-injected `RewardFn`,
  defaulting to the `wfcrl_reward` factory).
- `design/`: `Designer` type alias (a pure `(key, batch) -> layouts` callable,
  no `Protocol` — current designers are stateless closures), `SiteSpec`,
  `project_feasible` min-spacing projection, random/grid designers.
- `metrics.py`: rose-weighted power surface, AEP and wake loss.
- `viz/`: episode replay and flow pictures — `record.py` (`EpisodeRecord` +
  `.npz` save/load and the recorder), `plane.py` (horizontal/vertical slice
  framing over `physics.query_field`), `field.py` (per-frame hub-height wake
  fields, LRU-cached), `server.py` + `app.html` (stdlib HTTP server and the
  bundled canvas viewer, run via `python -m windrl_engine.viz episode.npz`).
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
- `logging/`: `Logger` protocol (`log_stat`/`log_config`/`stop`) plus its
  `wandb`/`console`/`null` implementations, so trainer code never imports
  wandb directly.
- `nn/`: equinox network modules for the PPO trainer — `mlp.py` (`MLP`, a
  trailing-axis feedforward net so `(envs, agents, feat)` batches need no
  vmap), `actor.py` (`Actor`, a per-agent scalar delta-yaw policy: a
  `distrax.Transformed` Normal squashed through tanh and scaled to
  `[-action_scale, action_scale]`), `critic.py` (`Critic`, a per-agent
  state-value head).
- `algo/ppo/`: IPPO scaffolding — `config.py` (`IPPOConfig`), `types.py`
  (`Transition`, `LearnerState` pytrees), `featurize.py` (`agent_features`,
  the raw `Observation` -> `NFEAT`-wide per-agent feature vector).
- `eval/`: `evaluator.py` (`evaluate`, a jitted fresh-reset deterministic
  rollout under `actor.mode` scoring `eval/mean_reward` for the runner to log).

## Outputs

Everything (checkpoints, logged layouts, wandb runs) defaults under the
gitignored `outputs/` at the repo root via `WindRlSettings`/`WIND_RL_WDIR`.
