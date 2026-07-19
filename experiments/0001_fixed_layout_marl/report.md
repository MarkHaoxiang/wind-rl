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
  CPU/GPU agnostic. `WANDB_MODE=disabled`.
- **Verdict (asserted in `run.py`, per variant).** Deterministic eval every
  iteration; a variant PASSes iff the mean of the last third of its evals
  strictly exceeds its first third. Each variant passes or fails independently;
  the run exits nonzero iff any variant fails its own baseline.

## Results

**BENCHMARK PASS** — both variants learn. Total wall-clock **~423 s (~7 min)**,
under the 15 min budget. Real run, `WANDB_MODE=disabled`, seed 0.

| variant | first window | last window |   delta | wall-clock |
| ------- | -----------: | ----------: | ------: | ---------: |
| `mlp`   |      32.2381 |     33.5124 | +1.2744 |    209.8 s |
| `gcn`   |      32.3501 |     33.2348 | +0.8847 |    212.8 s |

Both eval trajectories dip slightly in the first few iterations (initial
exploration off the zero-yaw init) then climb near-monotonically: `mlp` reaches
33.98 at the final iteration, `gcn` 33.37. Reference probes on this scenario
(fixed-yaw holds): zero-yaw ~= 30.6 per episode, best fixed upstream steering
(+25 deg) ~= 35.6 — both learned policies recover most of the available
wake-steering gain.

Per-variant checkpoints (policy/critic state_dicts + config) are written under
`WIND_RL_WDIR/0001_fixed_layout_marl_<variant>/`.

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
