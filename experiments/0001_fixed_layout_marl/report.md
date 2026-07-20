# 0001 — Fixed-layout MARL benchmark

## Hypothesis

On a **fixed** wind-farm layout, a set of MARL agent architectures — each a
shared-parameter policy/critic trained by the same MAPPO loop (SyncDataCollector
-> GAE -> clipped-PPO minibatch updates -> deterministic eval -> checkpoint) —
all *learn* wake steering under an identical PPO budget: each variant's
deterministic-eval mean episode reward (total farm power) rises above its own
early-training baseline. The benchmark tests learning per variant; it does
**not** claim to rank architectures at this scale.

## Setup

- **Scenario.** 3 turbines in a row at `[(252, 1000), (756, 1000), (1260, 1000)]`
  (504 m ~= 4 D spacing), map 2000x2000, `max_steps=20`. Wind is **fixed** at
  270 deg / 8 m/s (`scenario.fixed_wind_direction`). Fixed aligned wind is
  deliberate: turbines 2 and 3 sit squarely in turbine 1's wake, giving large,
  consistent wake-steering headroom and a noise-free deterministic eval. The
  layout is a config choice — swap in a real fixed farm via
  `wind_rl.scenario.real_farm_layout(<name>)` (matching `scenario.n_turbines`);
  the row is the default.
- **Variants (`conf/config.yaml`).** Each entry pairs a name with a typed `model`
  config; `kind` discriminates the model union, so the benchmark grows by
  appending to `variants`.
  - **`mlp`** — `MultiAgentMLP` actor/critic, shared params, `num_cells=64`,
    `depth=2`, `initial_std=0.3`.
  - **`gcn`** — dense-adjacency graph conv (KNN, torch-native), `hidden_dim=64`,
    `num_layers=2`, `initial_std=0.3`; the graph is rebuilt each forward from the
    turbine layout, giving structural permutation equivariance.
  - Both: yaw action is a per-step increment in [-5, +5] deg (`TanhNormal`);
    centralized critic emitting one `state_value` per agent.
- **PPO (identical across variants).** clip 0.2, gamma 0.99, lambda 0.95,
  entropy 0.0, advantage normalized, Adam lr 3e-4 with cosine decay to 1e-4,
  grad-clip 1.0, 8 epochs x 4 minibatches.
- **Budget.** 40 iterations x 1000 frames (40k env steps) per variant, seed 0,
  CPU/GPU agnostic. Logged online to wandb (`base.logging.use_wandb=true`,
  project `wind-rl`); the loop runs identically with wandb disabled.
- **Verdict (asserted in `run.py`, per variant).** Deterministic eval every
  iteration; a variant PASSes iff the mean of the last third of its evals
  strictly exceeds its first third. Each variant passes or fails independently;
  the run exits nonzero iff any variant fails its own baseline.

## Results

**BENCHMARK PASS** — both variants learn. Total wall-clock **~494 s (~8 min)**,
under budget. Real run, seed 0, wandb **online** to project `wind-rl`
(deterministic-eval reward = total farm power).

| variant | first window | last window |   delta | wall-clock | wandb                                                      |
| ------- | -----------: | ----------: | ------: | ---------: | --------------------------------------------------------- |
| `mlp`   |      32.2381 |     33.5124 | +1.2744 |    219.5 s | [o6sve6j7](https://wandb.ai/mark-haoxiang/wind-rl/runs/o6sve6j7) |
| `gcn`   |      32.3501 |     33.2348 | +0.8847 |    274.8 s | [bjxu76k9](https://wandb.ai/mark-haoxiang/wind-rl/runs/bjxu76k9) |

Both eval trajectories dip slightly in the first few iterations (initial
exploration off the zero-yaw init) then climb near-monotonically: `mlp` reaches
33.98 at the final iteration, `gcn` 33.37. Reference probes on this scenario
(fixed-yaw holds): zero-yaw ~= 30.6 per episode, best fixed upstream steering
(+25 deg) ~= 35.6 — both learned policies recover most of the available
wake-steering gain.

### Telemetry and framework health

The trainer logs ~34 namespaced metrics per iteration (`train/ loss/ optim/
time/ eval/ designer/`), a rendered eval-layout image, and an end-of-run
checkpoint wandb Artifact; the identical dict is returned in-process so the
verdict and tests need no wandb. Final-iteration health readout:

| metric                       |    mlp |    gcn |
| ---------------------------- | -----: | -----: |
| pre-clip grad norm           |   8.05 |   7.91 |
| clip fraction                |  0.015 |  0.005 |
| approx KL                    |  0.003 |  0.002 |
| critic explained variance    |  -0.01 |  -0.01 |
| policy entropy (base-normal) |   0.20 |   0.21 |
| action yaw std (deg)         |   1.27 |   1.31 |

PPO is stable and conservative — clip fraction < 2 % and approx-KL < 0.005 keep
every update well inside the trust region. Two caveats the telemetry exposes,
neither blocking (no tuning pass was needed): the pre-clip grad norm sits ~8x
above `max_grad_norm = 1.0`, so grad clipping is active on every step (raising
the clip or lowering lr is the lever if faster convergence is wanted); and
critic explained variance is ~0 — under fixed wind the episode return is nearly
constant, leaving little return variance for the value head to explain, yet
normalized GAE advantages still drive the policy to the wake-steering optimum.

### Scale probe (6-turbine row, `mlp`)

An exploratory `mlp` run on a 6-turbine westerly row (x = 252..2772 m at 504 m
spacing, map 3200x2000, 20 iters x 1000 frames) confirms the loop scales past
the 3-turbine benchmark:

| variant | first window | last window |   delta | wall-clock | wandb                                                      |
| ------- | -----------: | ----------: | ------: | ---------: | --------------------------------------------------------- |
| `mlp`   |      28.8248 |     32.0242 | +3.1994 |    164.3 s | [bsicnfgq](https://wandb.ai/mark-haoxiang/wind-rl/runs/bsicnfgq) |

More downstream turbines in wake give larger headroom and a larger,
near-monotonic delta (27.97 -> 32.26). Health metrics match the 3-turbine runs
(grad norm 7.98, clip fraction 0, KL ~0, explained variance ~0). PASS.

Per-variant checkpoints (policy/critic state_dicts + config) are written under
`WIND_RL_WDIR/0001_fixed_layout_marl_<variant>/`; the final checkpoint is also
uploaded as a wandb Artifact.

## Decision

The fixed-layout MARL benchmark is settled as a framework: it trains an
extensible set of architectures under one PPO budget on a shared fixed layout,
gates each on its own learning verdict, and tabulates the comparison. Both `mlp`
and `gcn` learn wake steering and PASS.

This run does **not** crown a winner. At smoke scale (3 turbines, 40k steps) the
`mlp` shows a larger windowed delta, but the KNN graph collapses to a nearly
fully-connected 3-node graph and the headroom is small — the gap is not a
meaningful architecture ranking. Separating the architectures needs a larger
budget and more turbines, where the GCN's permutation equivariance and locality
are expected to matter; scaling turbine count and adding further variants (still
fixed-layout) is the next step here. Co-design (per-episode designed layouts) is
a separate framework, starting at 0002+.

Caveat carried forward: the fixed-wind regime is what exposes a clean learning
signal at this scale. Under wfcrl's random wind a *wind-conditioned* policy is
required to beat zero-yaw and the average headroom is small — a stronger test
for later, ideally with a per-direction evaluation protocol.
