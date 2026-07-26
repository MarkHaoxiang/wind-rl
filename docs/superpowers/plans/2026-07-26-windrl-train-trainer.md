# windrl-train MARL Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Ownership:** Tasks marked **[AGENT]** are dispatched to subagents. Tasks
> marked **[OWNER]** are hand-written by the repo owner as a MARL learning
> exercise — the orchestrating Claude briefs the owner on concepts, the owner
> writes the code, tests gate it, and a reviewer agent bug-hunts before commit.
> Subagents MUST NOT implement [OWNER] task bodies (tests for them are fine).

**Goal:** An in-repo JAX MARL trainer (IPPO first; MAPPO/HAPPO next) in
`windrl-train`, per the spec at
`docs/superpowers/specs/2026-07-26-windrl-train-design.md`.

**Architecture:** Pure-functional anakin-style learner: env-step `lax.scan`
rollout → GAE → PPO epoch/minibatch updates, all inside one jitted
`update_fn(LearnerState) -> (LearnerState, metrics)`; a plain Python runner
loop drives updates/eval/logging with an injected `Logger`. PPO family shares
rollout+GAE+runner; each system supplies its update step.

**Tech Stack:** JAX, equinox, distrax, optax (windrl-train only), pydantic v2
configs, pytest. Engine: `windrl-engine` (pure JAX, FLORIS-verified).

## Global Constraints

- Python 3.12, one uv venv; `uv sync` after any dependency change; `uv.lock` committed.
- No torch, no torch_scatter/torch_cluster, no hydra, no flax. NN stack is equinox+distrax+optax.
- All configs are pydantic v2 via `windrl_train.config.Config` (`extra="forbid"`).
- No hardcoded paths — everything under `WindRlSettings` (`WIND_RL_*`).
- Style per `docs/coding-guidelines.md`: strict typing (jaxtyping shapes) is the documentation; comments only for genuine *why* or user-facing API.
- Gate for every commit: `uv run ruff check`, `ruff format --check`, `mypy`, `pytest -q` on the touched packages. Never `git add -A`; stage specific files.
- `docs/architecture.md` gets a one-line entry per new module, in the same task that creates the module.
- Multiple agents share this repo: never `git reset --hard` / `checkout -- .` / `clean -fd`. Keep engine diffs (Task 1) minimal and additive.
- wandb run names never encode the seed (seed goes in tags).

---

### Task 1: Engine pure functional API **[AGENT]**

**Files:**
- Modify: `packages/windrl-engine/src/windrl_engine/env/env.py`
- Modify: `packages/windrl-engine/src/windrl_engine/env/__init__.py` (export `EnvState`)
- Test: `packages/windrl-engine/tests/test_env_functional.py`
- Modify: `docs/architecture.md` (env bullet: mention public pure API)

**Interfaces:**
- Consumes: existing `_batched_reset` / `_batched_step` internals.
- Produces (public, scan-safe, pure):

```python
class EnvState(NamedTuple):
    farm: FarmState      # every leaf batched (envs, ...)
    layout: FarmLayout   # always per-env: leaves (envs, turbines)

# methods on BatchedWindFarmEnv (pure in their arguments; self holds only
# static config/turbine):
def reset_fn(self, key: Key[Array, ""], layouts: FarmLayout | None = None
             ) -> tuple[EnvState, Observation]: ...
def step_fn(self, state: EnvState, actions: Float[Array, "envs turbines"],
            key: Key[Array, ""]
            ) -> tuple[EnvState, Observation, Float[Array, "envs"], Bool[Array, "envs"]]: ...
```

Design notes for the implementer: `reset_fn` tiles a shared layout to
`(envs, turbines)` so `step_fn` always vmaps layout over axis 0 — this removes
the `per_env_layout` static flag from the public contract. Auto-reset keeps
using `state.layout`, so per-env layouts survive episode boundaries. The
stateful `reset`/`step`/`rollout` are reimplemented as thin shells that call
`reset_fn`/`step_fn` and stash `(state, obs)` on `self` — no behavior change.

- [ ] **Step 1: Write failing tests**

```python
# test_env_functional.py — key cases (write all four):
def test_step_fn_is_pure(env3):  # same (state, actions, key) twice -> identical pytrees
def test_scan_matches_stateful(env3):
    # 10 steps: lax.scan over env.step_fn with fixed per-step keys must produce
    # the same rewards as driving the stateful env.step with the same keys.
    # (Derive keys identically on both paths; if the stateful path's internal
    # key threading can't be replicated, instead assert scan(step_fn) equals a
    # hand-written python loop over step_fn — the contract under test is
    # scan-safety and purity, not stateful-path bit-equality.)
def test_reset_fn_tiles_shared_layout(env3):  # state.layout leaves have shape (n_envs, 3)
def test_per_env_layouts_survive_autoreset():
    # 2 envs, horizon=3, distinct per-env layouts; scan past truncation;
    # assert state.layout unchanged after auto-reset.
```

- [ ] **Step 2: Run tests, verify they fail** (`uv run pytest packages/windrl-engine/tests/test_env_functional.py -q` — AttributeError / import error expected)
- [ ] **Step 3: Implement `EnvState`, `reset_fn`, `step_fn`; rebase stateful wrappers onto them**
- [ ] **Step 4: Full engine test suite passes** (`uv run pytest packages/windrl-engine -q` — existing parity/invariant tests prove no behavior change)
- [ ] **Step 5: ruff/mypy gate, update `docs/architecture.md`, commit**

---

### Task 2: `logging/` package with injected `Logger` protocol **[AGENT]**

**Files:**
- Create: `packages/windrl-train/src/windrl_train/logging/__init__.py` (protocol + re-exports)
- Create: `.../logging/wandb.py` (move existing `WandbLogger` here, unchanged API)
- Create: `.../logging/console.py`, `.../logging/null.py`
- Delete: `packages/windrl-train/src/windrl_train/logging.py`
- Test: `packages/windrl-train/tests/test_logging.py`
- Modify: `docs/architecture.md`

**Interfaces:**
- Produces:

```python
@runtime_checkable
class Logger(Protocol):
    def log_stat(self, key: str, value: float, step: int, event: str) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...
    def stop(self) -> None: ...

class ConsoleLogger:  # prints "step=N event/key=value"
class NullLogger:     # no-ops; also records calls in .stats list for test assertions
```

- Consumes: existing `WandbLogger` (keeps `log_stat`/`log_config`/`stop`; moves module, no signature change). `wandb` is imported **only** inside `logging/wandb.py`.

- [ ] **Step 1: Failing tests** — `isinstance(NullLogger(), Logger)` etc. for all three; `NullLogger.stats` records `(key, value, step, event)`; `from windrl_train.logging import Logger, NullLogger, ConsoleLogger, WandbLogger` all import without initializing wandb.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement package; grep repo for `from windrl_train.logging import` / `windrl_train.logging` and fix any existing imports**
- [ ] **Step 4: Tests + full windrl-train suite pass**
- [ ] **Step 5: ruff/mypy gate, architecture.md, commit**

---

### Task 3: deps + `nn/` package **[AGENT]**

**Files:**
- Modify: `packages/windrl-train/pyproject.toml` (add `equinox>=0.13`, `distrax>=0.1.5`, `optax>=0.2.8`), run `uv sync`, commit `uv.lock`
- Create: `.../windrl_train/nn/__init__.py`, `nn/mlp.py`, `nn/actor.py`, `nn/critic.py`
- Test: `packages/windrl-train/tests/test_nn.py`
- Modify: `docs/architecture.md`

**Interfaces (one module per network class):**

```python
# nn/mlp.py — operates on the TRAILING axis (works on (..., feat) batches, no vmap needed)
class MLP(eqx.Module):
    def __init__(self, in_size: int, out_size: int, width: int, depth: int,
                 *, key: Key[Array, ""]) -> None: ...
    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]: ...
    # layers: x @ W + b with jax.nn.silu between; depth = number of hidden layers

# nn/actor.py — per-agent scalar delta-yaw policy (shared params, applied over agent axis)
class Actor(eqx.Module):
    torso: MLP                 # feat -> 1 (mean)
    log_std: Float[Array, ""]  # state-independent, init log(0.5)
    action_scale: float        # = env yaw_step; static field
    def __call__(self, feats: Float[Array, "... feat"]) -> distrax.Distribution:
        # distrax.Transformed(Normal(mu, exp(log_std)),
        #                     Chain([ScalarAffine(0, action_scale), Tanh()]))
        # batch shape (...,): per-agent independent actions
    def mode(self, feats: Float[Array, "... feat"]) -> Float[Array, "..."]:
        # deterministic eval action: tanh(mu) * action_scale

# nn/critic.py
class Critic(eqx.Module):
    torso: MLP  # feat -> 1
    def __call__(self, feats: Float[Array, "... feat"]) -> Float[Array, "..."]: ...
```

- [ ] **Step 1: Failing tests** — shapes on `(envs, agents, feat)` inputs; `Actor(...)(feats).sample(seed=k)` shape `(envs, agents)` with values strictly inside `[-scale, scale]`; `log_prob(sample)` finite; `mode` inside bounds; `Critic` returns `(envs, agents)`; modules are pytrees (`jax.tree.leaves` non-empty, `eqx.filter_jit(lambda m, x: m(x))` works).
- [ ] **Step 2: Run, verify fail** (imports fail until deps added — add deps first, then tests fail on missing modules)
- [ ] **Step 3: Implement**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Gate, architecture.md, commit** (pyproject+lock staged in this commit)

---

### Task 4: `algo/ppo/` config, types, featurize **[AGENT]**

**Files:**
- Create: `.../windrl_train/algo/__init__.py`, `algo/ppo/__init__.py`, `algo/ppo/config.py`, `algo/ppo/types.py`, `algo/ppo/featurize.py`
- Test: `packages/windrl-train/tests/test_ppo_scaffolding.py`
- Modify: `docs/architecture.md`

**Interfaces:**

```python
# config.py
class IPPOConfig(Config):  # windrl_train.config.Config base (pydantic, extra="forbid")
    env: WindFarmEnvConfig
    total_timesteps: int = 1_000_000     # env steps summed over envs
    rollout_length: int = 128
    ppo_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.5
    width: int = 64
    depth: int = 2
    eval_every_updates: int = 10
    eval_steps: int = 512          # deterministic steps per eval (env auto-resets)

# types.py
class Transition(NamedTuple):
    obs: Float[Array, "envs agents feat"]
    action: Float[Array, "envs agents"]
    log_prob: Float[Array, "envs agents"]
    value: Float[Array, "envs agents"]
    reward: Float[Array, "envs"]   # shared cooperative reward
    done: Bool[Array, "envs"]      # truncation (env auto-resets)

class LearnerState(NamedTuple):
    actor: Actor
    critic: Critic
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    env_state: EnvState
    obs: Observation               # raw env obs (featurized at use)
    key: Key[Array, ""]
    timestep: Int[Array, ""]       # env steps so far (envs * steps)

# featurize.py — trainer-side view of the raw Observation; NFEAT = 7
def agent_features(obs: Observation) -> Float[Array, "envs agents 7"]:
    # per turbine i: [yaw_i/40, ws_i/28, sin(wd_i°), cos(wd_i°),
    #                 fw_speed/28, sin(fw_dir°), cos(fw_dir°)]
    # (angles converted deg->rad; freewind broadcast to every agent)
NFEAT: int = 7
```

- [ ] **Step 1: Failing tests** — `IPPOConfig(env=WindFarmEnvConfig())` builds; typo'd field raises `ValidationError`; `agent_features` output shape/(finite, |sin/cos|<=1) on a real `env.reset_fn` observation; `Transition`/`LearnerState` construct and are pytrees.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Gate, architecture.md, commit**

---

### Task 5: `eval/evaluator.py` **[AGENT]**

**Files:**
- Create: `.../windrl_train/eval/__init__.py`, `eval/evaluator.py`
- Test: `packages/windrl-train/tests/test_evaluator.py`
- Modify: `docs/architecture.md`

**Interfaces:**
- Consumes: `env.reset_fn`/`env.step_fn` (Task 1), `Actor.mode` (Task 3), `agent_features` (Task 4).
- Produces:

```python
def evaluate(env: BatchedWindFarmEnv, actor: Actor, key: Key[Array, ""],
             n_steps: int) -> dict[str, Float[Array, ""]]:
    # fresh reset_fn, one lax.scan of n_steps deterministic steps
    # (action = actor.mode(agent_features(obs))), returns
    # {"eval/mean_reward": mean over (steps, envs)}
    # jitted internally; n_steps static.
```

Deterministic policy + fixed wind per episode makes this the clean signal
`verdict.windowed_delta` scores; the runner logs it with event="eval".

- [ ] **Step 1: Failing test** — 3-turbine env (`n_envs=2`, `horizon=8`), random-init `Actor`: returns finite scalar; same key twice -> identical result (determinism); different actor params -> (almost surely) different result.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Gate, architecture.md, commit**

---

### Task 6: `algo/ppo/rollout.py` **[OWNER]**

**Files:**
- Create: `.../algo/ppo/rollout.py` (owner-written)
- Test: `packages/windrl-train/tests/test_rollout.py` (agent-written, BEFORE owner starts)

**Interfaces:**
- Consumes: `env.step_fn`, `agent_features`, `Actor`/`Critic`, `Transition`, `LearnerState`.
- Produces:

```python
def collect_rollout(state: LearnerState, env: BatchedWindFarmEnv, n_steps: int
                    ) -> tuple[LearnerState, Transition]:
    # lax.scan over env.step_fn; Transition leaves stacked to (steps, envs, ...);
    # returned LearnerState has advanced env_state/obs/key/timestep and
    # UNCHANGED params/opt states.
```

- [ ] **Step 1 [AGENT]: Write failing tests** — leaf shapes `(n_steps, envs, agents)` / `(n_steps, envs)`; params unchanged (`jax.tree.all(jax.tree.map(lambda a,b: (a==b).all(), ...)`) on actor leaves); `timestep` advanced by `n_steps * n_envs`; `log_prob` equals `actor(obs_feats).log_prob(action)` recomputed from the stored transition; works under `eqx.filter_jit`.
- [ ] **Step 2: Concept briefing** — Claude walks the owner through: scan carry design, why obs stored is the *pre-step* obs, key splitting discipline (act key vs step key), where `value` is computed.
- [ ] **Step 3 [OWNER]: Implement until tests pass**
- [ ] **Step 4 [AGENT]: Bug-hunt review** (off-by-one obs/reward alignment, key reuse, tracer leaks) — findings fixed by owner.
- [ ] **Step 5: Gate, commit**

---

### Task 7: `algo/ppo/gae.py` **[OWNER]**

**Files:**
- Create: `.../algo/ppo/gae.py` (owner-written)
- Test: `packages/windrl-train/tests/test_gae.py` (agent-written first)

**Interfaces:**

```python
def gae(traj: Transition, last_value: Float[Array, "envs agents"],
        gamma: float, gae_lambda: float
        ) -> tuple[Float[Array, "steps envs agents"], Float[Array, "steps envs agents"]]:
    # returns (advantages, value_targets); targets = advantages + traj.value.
    # Shared reward broadcast over the agent axis. Convention (matches Mava):
    #   delta_t = r_t + gamma * (1-d_t) * v_{t+1} - v_t
    #   A_t     = delta_t + gamma * lam * (1-d_t) * A_{t+1}
    # d_t is traj.done (truncation treated as termination — discussed in briefing).
```

- [ ] **Step 1 [AGENT]: Failing tests with hand-computed numbers** (gamma=0.9, lam=0.8, T=3, 1 env, 1 agent; v=[1,2,3], last_value=4, r=[1,1,1]):
  - no dones: advantages == [3.85344, 2.852, 1.6] (rtol 1e-6); targets == adv + v
  - done=[F,T,F]: advantages == [1.08, -1.0, 1.6]
  - shared-reward broadcast: 2 agents with different values get different advantages from the same reward
  - implemented as `lax.scan` in reverse (test: jits, and handles T=1)
- [ ] **Step 2: Concept briefing** — TD residuals, bias/variance of lambda, why truncation-as-termination is a (deliberate, Mava-matching) bias under time limits, reverse scan mechanics.
- [ ] **Step 3 [OWNER]: Implement until tests pass**
- [ ] **Step 4 [AGENT]: Bug-hunt review** (reversed-scan direction, done indexing, broadcast bugs)
- [ ] **Step 5: Gate, commit**

---

### Task 8: `algo/ppo/ippo.py` — losses + update step **[OWNER]**

**Files:**
- Create: `.../algo/ppo/ippo.py` (owner-written)
- Test: `packages/windrl-train/tests/test_ippo.py` (agent-written first)

**Interfaces:**
- Consumes: everything above.
- Produces:

```python
def make_learner(config: IPPOConfig, env: BatchedWindFarmEnv, key: Key[Array, ""]
                 ) -> tuple[LearnerState, Callable[[LearnerState], tuple[LearnerState, dict[str, Array]]]]:
    # update_fn (pure, filter_jit-able):
    #   collect_rollout -> bootstrap last_value -> gae -> normalize advantages
    #   -> epoch scan (ppo_epochs) over minibatch scan (num_minibatches):
    #        permute (steps*envs) axis, reshape; agents axis stays whole
    #        actor loss: clipped surrogate (per-agent ratios) - ent_coef * entropy
    #        critic loss: 0.5 * MSE(value, targets)
    #        separate optax.adam chains with clip_by_global_norm(max_grad_norm)
    #   -> metrics dict: actor_loss, critic_loss, entropy, approx_kl, mean_reward
```

- [ ] **Step 1 [AGENT]: Failing tests**
  - `update_fn` runs under `eqx.filter_jit` on a 3-turbine env and returns finite metrics; actor and critic params change
  - zero-advantage invariance: monkeypatched/zeroed advantages with `ent_coef=0` -> actor params unchanged (clip surrogate has zero gradient at ratio 1 with zero advantage)
  - critic regression: repeated updates on a frozen trajectory reduce critic loss
  - `approx_kl >= 0` and finite
- [ ] **Step 2: Concept briefing** — importance ratios & clipping geometry, why advantages are normalized per batch, epoch/minibatch scan structure, `eqx.filter_value_and_grad` + optax wiring, what IPPO *doesn't* share (critic sees local features only).
- [ ] **Step 3 [OWNER]: Implement until tests pass**
- [ ] **Step 4 [AGENT]: Bug-hunt review** (ratio sign errors, stop-gradient on targets, minibatch permutation correlated across epochs, log_prob summed vs per-agent)
- [ ] **Step 5: Gate, commit**

---

### Task 9: `algo/ppo/runner.py` **[OWNER + Claude]**

**Files:**
- Create: `.../algo/ppo/runner.py`
- Test: `packages/windrl-train/tests/test_runner.py` (agent-written first)

**Interfaces:**

```python
def run(config: IPPOConfig, logger: Logger, key: Key[Array, ""]) -> list[float]:
    # builds env from config.env, make_learner, filter_jits update_fn once;
    # python loop: for each update -> update_fn; every eval_every_updates ->
    # evaluate(...) with a fixed eval key, logger.log_stat(..., event="eval"),
    # train metrics logged event="train" at state.timestep.
    # checkpoints: eqx.tree_serialise_leaves((actor, critic)) to
    # WindRlSettings().resolved_wdir / <run subdir> at each eval.
    # returns the eval-reward series (the input to verdict.windowed_delta).
```

- [ ] **Step 1 [AGENT]: Failing test** — tiny config (3 turbines, `n_envs=2`, `rollout_length=8`, 3 updates, eval every 1) with `NullLogger`: returns 3-and-only-3 eval values, NullLogger recorded both train and eval events, checkpoint file exists under a tmp `WIND_RL_WDIR`.
- [ ] **Step 2 [OWNER + Claude]: Implement together** (jit-once discipline, logger boundary, no paths outside settings)
- [ ] **Step 3 [AGENT]: Bug-hunt review**
- [ ] **Step 4: Gate, commit**

---

### Task 10: IPPO smoke test, full review, docs **[AGENT]**

**Files:**
- Test: `packages/windrl-train/tests/test_ippo_smoke.py`
- Modify: `docs/architecture.md` (windrl-train section rewrite: logging/, nn/, algo/ppo/, eval/)
- Delete: `docs/superpowers/plans/2026-07-26-windrl-train-trainer.md` (this plan — once green)

**Steps:**
- [ ] **Step 1: Smoke test** — `turb3_row1`, CPU, small budget (e.g. `n_envs=8`, `rollout_length=64`, ~30 updates, eval every 3): assert `windowed_delta(evals).delta > 0`. Budget tuned so the suite stays fast; mark `@pytest.mark.slow` if >~90s and wire the marker into CI's full run.
- [ ] **Step 2: Whole-module bug-hunt review** of `algo/ppo/` + `nn/` by a fresh reviewer agent (adversarial: alignment, keys, broadcasting, silent-NaN paths); owner fixes findings.
- [ ] **Step 3: Full gate on both packages; update architecture.md; delete this plan; commit**

---

## M3 / M4 outline (planned in detail after M2 ships)

- **M3 — MAPPO** [OWNER]: `featurize.global_features(obs) -> (envs, gfeat)`
  (concat of all agents' features + freewind, optional layout coords flag);
  `mappo.py` update step where the critic consumes global features (per-agent
  value heads over shared global input); config `MAPPOConfig`. Reuses rollout,
  gae, runner untouched — that is the test of the Task-6/7/9 boundary.
- **M4 — HAPPO** [OWNER]: stacked per-agent actor params (`eqx` module with a
  leading agent axis via `jax.vmap` over init keys), sequential agent update
  loop (`lax.scan` over a permuted agent order) with the compounding
  advantage-ratio factor; `HAPPOConfig`. Rollout/gae unchanged.
- **Later — MAT, HASAC**: sibling packages `algo/mat/`, `algo/hasac/`
  (HASAC brings flashbax + orbax); planned separately.
