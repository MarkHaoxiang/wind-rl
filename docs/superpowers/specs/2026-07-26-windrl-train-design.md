# windrl-train MARL trainer — design

**Status:** temporary planning doc (repo convention: delete once implemented;
durable state lives in code, tests, and `docs/architecture.md`).

## Goal

An in-repo, from-scratch JAX MARL trainer in `windrl-train`, replacing nothing
(the old trainer is already gone) and depending on Mava only as a reading
reference. Algorithm ladder: **IPPO → MAPPO → HAPPO**, later **MAT** and
**HASAC** as sibling systems. The owner hand-writes every learner (rollout
scan, GAE, losses, update steps) as a MARL learning exercise; agents build the
surrounding plumbing and review the hand-written code for bugs.

## Structure decision (hybrid)

The PPO family shares **rollout collection + GAE + the runner**; each system
supplies its **update step**. That boundary survives HAPPO (whose differences —
sequential per-agent updates, compounding importance ratio, per-agent params —
live entirely inside the update). MAT (autoregressive action decoding) and
HASAC (off-policy: replay buffer, target networks) do not fit that skeleton
and land later as sibling packages with no shared base class. No inheritance
hierarchies anywhere.

## Package layout

```
windrl_train/
  config.py, settings.py, verdict.py    # existing harness, unchanged
  logging/
    __init__.py       # Logger Protocol: log(metrics, step), finish()
    wandb.py          # WandbLogger moves here, implements Logger
    console.py        # stdout logger for local runs
    null.py           # no-op for tests
  nn/                 # one module per network class (settlrl convention)
    mlp.py            # MLP torso
    actor.py          # Actor: torso + action head -> distrax distribution
    critic.py         # Critic: torso -> value
    # later research-ladder rungs: set_transformer.py, so2_equiformer.py, ...
  algo/
    ppo/
      config.py       # IPPOConfig / MAPPOConfig / HAPPOConfig (pydantic Config)
      types.py        # Transition, LearnerState, metrics NamedTuples
      rollout.py      # env-step lax.scan collecting Transitions      [OWNER]
      gae.py          # advantage computation                         [OWNER]
      ippo.py         # losses + update step                          [OWNER]
      mappo.py        # losses + update step                          [OWNER]
      happo.py        # losses + update step                          [OWNER]
      runner.py       # jit boundary, update/eval/log/ckpt loop  [owner+Claude]
    mat/              # later, sibling
    hasac/            # later, sibling
  eval/
    evaluator.py      # deterministic eval episodes -> metrics
```

`verdict.py` stays top-level (experiments import it independently of any
algorithm). Update `docs/architecture.md` (one line per module) as pieces land.

## Engine contract (coordinated change in windrl-engine)

Promote the existing pure internals to public API on `BatchedWindFarmEnv`:

- `reset_fn(key, layouts=None) -> (EnvState, Observation)`
- `step_fn(state, actions, key) -> (EnvState, Observation, reward, truncated)`

Pure, jit/scan-safe, batched over envs, auto-reset preserved. The stateful
`reset/step/rollout` become thin shells over these. One focused PR,
coordinated with the agent currently working on the engine.

## Learner shape

Each system module exports
`make_learner(config, env, networks) -> (LearnerState, update_fn)` with
`update_fn(LearnerState) -> (LearnerState, metrics)` doing: rollout scan
(shared) → GAE (shared) → system-specific epoch/minibatch scans and losses.
`update_fn` is pure; metrics are returned, never logged from inside jit.

- **Jit topology** (single GPU, no pmap): `jit(update_fn)` once; `runner.py`
  is a plain Python loop alternating updates / eval / log / checkpoint. No
  Mava-style updates-per-eval scan wrapper unless profiling demands it.
- **Parameter sharing:** IPPO/MAPPO use one shared actor and critic applied
  over the agent axis. Param handling must not preclude a leading agent axis
  (vmap over stacked per-agent params) — HAPPO requires unshared actors.
- **MAPPO centralized critic input** is built in the trainer, not the engine:
  concatenation of all agents' features + `freewind` (with a frozen layout
  that is the global state); layout coordinates as an optional flag for
  multi-layout training later.
- Reward is per-env and shared (fully cooperative); truncation-only episodes
  (no termination) with in-env auto-reset.

## Networks

**equinox + distrax + optax** (new deps of windrl-train only; engine
untouched). Chosen for cross-project consistency with catan-engine/settlrl,
which uses equinox with per-class `nn/` modules. Equinox modules are pytrees,
so policies sit directly in scan carries and `LearnerState`. Action head
follows `env.control_mode`: tanh-squashed Gaussian scaled to
`[-yaw_step, yaw_step]` (continuous) or categorical over 3 actions (discrete).

## Config, logging, evaluation

- Pydantic `Config` subclass per system (`extra="forbid"`, repo convention;
  no hydra): rollout_length, num_envs, epochs, minibatches, gamma, gae_lambda,
  clip_eps, ent_coef, vf_coef, lr, max_grad_norm, total_timesteps, eval
  cadence.
- **Logging is dependency-injected:** systems never import a concrete logger.
  The runner takes `logger: Logger`; wandb is imported in exactly one file;
  tests inject `NullLogger`.
- Evaluator runs deterministic-action episodes on held-out keys, returns mean
  power/reward metrics; `verdict.windowed_delta` remains the learning-signal
  gate experiments assert on.

## Testing & review loop

Agents write: GAE unit tests against hand-computed references, rollout
shape/dtype tests, and an IPPO smoke test (positive `windowed_delta` on
`turb3_row1` in a short run). Every owner-written milestone gets a dedicated
bug-hunt review pass by a reviewer agent before moving on.

## Milestones

1. **M0 — engine pure API**: spec handed to / coordinated with the engine
   agent; trainer work can start against the private fns and swap imports.
2. **M1 — scaffolding** (agents): `logging/`, `nn/`, `algo/ppo/{config,types}`,
   `eval/evaluator.py`, deps added, tests for the scaffolding.
3. **M2 — IPPO** (owner, guided): rollout → GAE → losses → update → runner;
   review pass; smoke test green.
4. **M3 — MAPPO** (owner): centralized critic input; review pass.
5. **M4 — HAPPO** (owner): per-agent params, sequential update; review pass.
6. **Later — MAT, HASAC** as sibling packages (HASAC brings flashbax/orbax).
