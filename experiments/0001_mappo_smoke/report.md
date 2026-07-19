# 0001 — MAPPO smoke (walking skeleton)

## Hypothesis

The MAPPO training loop (SyncDataCollector -> GAE -> clipped-PPO minibatch
updates -> deterministic eval -> checkpoint) is wired correctly end to end, and
on a 3-turbine FLORIS farm the policy *learns*: deterministic-eval mean episode
reward (total farm power) rises over a short run. This is the M1 walking
skeleton — plumbing plus a real learning signal — not a power-capture claim.

## Setup

- **Scenario.** 3 turbines in a row at `[(252, 1000), (756, 1000), (1260, 1000)]`
  (504 m ~= 4 D spacing), map 2000x2000, `max_steps=20`. Wind is **fixed** at
  270 deg / 8 m/s (`scenario.fixed_wind_direction`). Fixed aligned wind is
  deliberate: turbines 2 and 3 sit squarely in turbine 1's wake, giving large,
  consistent wake-steering headroom, and it makes the deterministic eval
  noise-free. Under wfcrl's default *random* wind (direction ~N(270, 20 deg)) a
  fixed yaw offset helps only near 270 deg and hurts off-axis, so zero-yaw (the
  policy init) is already near-optimal on average and there is little to learn —
  see Decision.
- **Model.** MLP actor/critic, shared params, `num_cells=64`, `depth=2`,
  `initial_std=0.3`. Yaw action is a per-step increment in [-5, +5] deg
  (`TanhNormal`); centralized critic.
- **PPO.** clip 0.2, gamma 0.99, lambda 0.95, entropy 0.0, advantage normalized,
  Adam lr 3e-4 with cosine decay to 1e-4, grad-clip 1.0, 8 epochs x 4
  minibatches.
- **Budget.** 40 iterations x 1000 frames (40k env steps), seed 0, CPU/GPU
  agnostic. `WANDB_MODE=disabled`.
- **Verdict (asserted in `run.py`).** Deterministic eval every iteration; PASS
  iff the mean of the last third of evals strictly exceeds the first third.
  (Eval is deterministic here, but the windowed comparison also smooths the mild
  jitter from training stochasticity.)

## Results

- **PASS.** Wall-clock **208 s** (~3.5 min), well under the 10 min budget.
- Eval reward climbs smoothly and near-monotonically: **first iteration 32.05 ->
  last iteration 33.98** (+6.0%).
- Windowed verdict: **first-third mean 32.24 -> last-third mean 33.51,
  delta +1.27** (strictly positive -> PASS).
- Reference points (fixed-yaw hold probes, same scenario): zero-yaw ~= 30.6
  per episode, best fixed upstream steering (+25 deg) ~= 35.6. The learned policy
  (~= 34.0) recovers most of the available wake-steering gain.
- Checkpoints (policy/critic state_dicts + config) written under
  `WIND_RL_WDIR/0001_mappo_smoke/` and confirmed reloadable (`tests/rl`).

## Decision

The MAPPO walking skeleton works: the loop trains, the policy learns to steer
wakes, the verdict gate passes, checkpoints round-trip. M1 plumbing is settled;
T5 (GNN policy) and larger turbine counts can build on this trainer.

Caveat for later work: the fixed-wind regime was necessary to expose a clean
learning signal at this scale. Under wfcrl's random wind, a *state-conditioned*
(wind-aware) policy is required to beat zero-yaw, and the average headroom is
small — a stronger test for the GNN/architecture experiments (T5+), ideally with
a wind-conditioned or per-direction evaluation protocol rather than a single
fixed direction.
