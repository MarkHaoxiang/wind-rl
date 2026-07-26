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
verified live against FLORIS 4.6.6 (a pinned runtime dependency; WFCRL is
not) at rtol 1e-12 for u/turbulence-intensity/power and rtol 1e-9 (+atol) for
the near-zero v/w transverse components;
pure-functional (`NamedTuple` PyTrees, single-farm cores, batched via
`jit(vmap)`), layered `farm` -> `physics` -> `env`, with `viz` sitting above
both env and physics (plus `design` and `metrics`), each layer importing only
from lower ones.

- `farm/`: `turbine.py` (`TurbineSpec`, NREL-5MW read from FLORIS's packaged
  `nrel_5MW.yaml` at import, without importing floris itself), `layout.py`
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
- `design/`: `Designer`, a `@runtime_checkable` `Protocol` for a pure
  `(key, batch) -> layouts` callable, `SiteSpec`, `project_feasible`
  min-spacing projection, and the `fixed`/`random_uniform` designers.
- `metrics.py`: rose-weighted power surface, AEP and wake loss.
- `viz/`: episode replay and flow pictures — `record.py` (`EpisodeRecord` +
  `.npz` save/load and the recorder), `plane.py` (horizontal/vertical slice
  framing over `physics.query_field`), `field.py` (per-frame hub-height wake
  fields, LRU-cached), `server.py` + `app.html` (stdlib HTTP server and the
  bundled canvas viewer, run via `python -m windrl_engine.viz episode.npz`).
- `tests/`: live parity against FLORIS 4.6.6, physics invariants, and a
  beartype import hook that runtime-checks jaxtyping shape annotations across
  `farm/`, `physics/`, `design/` and `metrics.py` (`env` batches those same
  un-batched aliases under `vmap` and is excluded).

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
