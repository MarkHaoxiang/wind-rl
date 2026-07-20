# 0001_wfcrl_baseline

## Hypothesis

Our torchrl MAPPO stack, given the *official* WFCRL NeurIPS-2024 training budget
and setup (not our previous under-powered runs), learns yaw-based wake steering
on the paper's Scenario I benchmark: on `Dec_Turb3_Row1_Floris` (3 turbines) and
`Dec_Ablaincourt_Floris` (7 turbines) it should raise total farm power ~+20% over
the zero-yaw greedy baseline, matching the paper's reported MAPPO results. This
is the corrective for the earlier flat-reward runs, which failed for budget /
wind-variance / reward-scale reasons rather than an algorithmic defect.

## Setup

Constant wind (`wind_speed=8`, `wind_direction=270`), yaw-only continuous control
(dyaw in [-5, 5] deg/step), episode length 150, wfcrl default reward with
`load_coef=1` (power/u_inf^3 - load penalty). Two farms, both fetched from the
wfcrl package and translated in-map by `resolve_real_farm` (no hardcoded
coordinates):

| variant       | turbines | layout                    | total frames      |
|---------------|----------|---------------------------|-------------------|
| `turb3_row1`  | 3        | `Turb3_Row1` (4D row)     | 200,704 (98x2048) |
| `ablaincourt` | 7        | `Ablaincourt` (real farm) | 200,704 (98x2048) |

PPO matches the official argparse defaults (paper Table 5): 2048 frames/update,
32 minibatches (size 64), 10 epochs, lr 3e-4 **linearly annealed to 0**,
gamma=0.99, GAE lambda=0.95, clip 0.2, **clipped value loss**, ent_coef 0.0,
vf_coef 0.5, max_grad_norm 0.5, Adam eps 1e-5, advantage normalisation on, and
**per-rollout reward standardisation** (rewards recomputed to zero-mean/unit-std
over each 2048 batch before GAE). Network: MLP (64, 64) Tanh, state-independent
learned log_std, deterministic eval = distribution mode. Eval is one
deterministic episode every 5 updates; the zero-yaw greedy episode power is
measured once at startup so gains read as percentages.

### Deviations from the official setup (recorded, all deliberate)

- **Seeds.** The paper uses 5 seeds (0-4); we run **2** (0, 1) per the owner's
  request to avoid many repeats.
- **Observation featurisation.** The official agents consume raw (local wind
  speed, direction, own yaw), min-max normalised. We keep our engineered
  per-agent features (cos/sin wind, yaw in radians, map-normalised layout). Same
  information, different encoding; a fair architecture-agnostic comparison of the
  learning dynamics, not a bit-exact port.
- **Ablaincourt budget.** The paper reports 1-2M frames for Ablaincourt (with
  convergence well before that). We run **2e5** and rely on the streamed eval
  curves + checkpoints to extend later if a run is still rising at the budget.
- **std parameterisation.** The official policy uses `std = exp(log_std)` with
  `log_std=0` (std=1). Ours applies torchrl's `biased_softplus`
  (`initial_std=1.0`) under a `TanhNormal` over the bounded dyaw action - initial
  scale ~1 but not identical.
- **Actuation budget.** wfcrl silently zeroes a turbine's yaw action once its
  cumulative actuation exceeds ~10% of the horizon. The official benchmark trains
  through the same constraint, so we leave it in place;
  `train/action_yaw_abs_mean` is logged so it stays observable.

## Results

Runs pending. The framework is built, smoke-tested end-to-end (both variants,
`WIND_RL_WANDB_MODE=disabled`, 2 iterations), and all four repo checks are green;
the full 2-seed runs have not been launched.

Paper reference targets to compare against (MAPPO, FLORIS Scenario I):

| variant       | score (paper)       | episode power (paper)  | baseline power |
|---------------|---------------------|------------------------|----------------|
| `turb3_row1`  | ~= 237.7            | ~= 2.75 MWh (from ~2.3)| ~2.3 MWh       |
| `ablaincourt` | ~= 351.7 (from ~290)| ~= 9.1 MWh (from ~7.5) | ~7.5 MWh       |

## Decision

Verdict gates (asserted in `run.py`, per variant across all seeds):

- **(a) steering:** final eval episode power >= +10% over the zero-yaw greedy
  baseline (`eval/power_gain >= 0.10`).
- **(b) learning:** final-third mean eval score >= first-third mean x 1.05.

A variant PASSes iff every seed clears both. Launch, then fill Results with the
actual score / episode power / power gain per variant alongside the paper targets
above, and record the concluded verdict in `JOURNAL.md` (owner-managed).

### Launch commands (per variant, online wandb, one process per farm)

    WIND_RL_WANDB_MODE=online uv run python experiments/0001_wfcrl_baseline/run.py config=turb3_row1
    WIND_RL_WANDB_MODE=online uv run python experiments/0001_wfcrl_baseline/run.py config=ablaincourt
