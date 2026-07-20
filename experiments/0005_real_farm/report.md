# 0005 — First training on a real wind-farm layout

## Hypothesis

The MAPPO co-design loop, proven at smoke scale on synthetic 3–8 turbine rows
(0001, 0003), **runs end-to-end on a REAL wind-farm layout at production scale**
(Ormonde, 30 turbines; HornsRev1, 80 turbines) and produces finite, well-behaved
telemetry — the capability the paper's "real farms up to 92 turbines" framing
needs. Secondarily, this is the first production use of the parallel FLORIS
collector (`TrainingConfig.n_envs`, commit 8a30157): auto-resolved parallel
collection should give a large collect-time speedup over the serial path. Whether
the policy *learns* wake steering at these scales under a short fixed-wind budget
is reported honestly, not assumed — partial or flat learning is plausible and is
not the framework's gate.

## Setup

- **Farms.** Real wfcrl `data_cases` fetched via
  `wind_rl.scenario.real_farm_layout`: **Ormonde** (30 turbines, primary) and
  **HornsRev1** (80 turbines, feasibility smoke). Real farms carry their own
  metre-scale coordinate frames — HornsRev1's y runs down to −1947 m. Wake
  physics depend only on *relative* turbine positions, so `run.py` translates the
  raw layout so its bounding-box corner sits at `(margin, margin)` (`margin=500`
  m) and derives the scenario map bounds from `bbox + margin` on every side. That
  keeps every coordinate positive and in-map — what the mlp position
  normalisation (`x / map_x_length * 2 − 1`), the renderer axes (`xlim 0..map_x`),
  and the `layout` observation Box (`low=0`) all assume. Ormonde resolves to a
  5164 × 4978 m map, HornsRev1 to 6512 × 4893 m; the layout enters via the
  fixed-layout path (`layout=`), so there is no per-episode designer and (fixed
  layout) resets do not rebuild the MDP.
- **Wind.** Fixed at 270° / 8 m/s (`fixed_wind_direction`): deterministic eval,
  large steering headroom, matches 0001's regime. Varied wind is a follow-up
  (needs a wind-conditioned policy and a per-direction eval protocol).
- **Variants.** `set_transformer` (0003's benchmark front-runner: pre-LN MHSA
  over turbine tokens, wind-frame canonicalisation, `embed_dim=64`, `num_heads=4`,
  `num_layers=2`) and `mlp` (`MultiAgentMLP`, `num_cells=64`, `depth=2`) for
  reference. Both `initial_std=0.3`, yaw a per-step increment in [−5, +5]°.
- **PPO (identical across variants).** clip 0.2, γ 0.99, λ 0.95, entropy 0.0,
  advantage normalised, Adam lr 3e-4 → cosine 1e-4, grad-clip 1.0, 8 epochs × 4
  minibatches.
- **Budget.** Ormonde: 40 iterations × 1000 frames (40k env steps) per variant;
  HornsRev1 smoke: 5 iterations × 1000 frames. seed 0, **CPU** (FLORIS is
  CPU-bound; the models are tiny). `n_envs=null` → auto **20** parallel FLORIS
  collectors. `RewardNormalisation` runs in identity mode for these scenarios (no
  precomputed stats — expected, the normaliser is a no-op here). Online to wandb
  project `wind-rl`.
- **Verdict (asserted in `run.py`).** CAPABILITY (hard gate → exit code): every
  variant's run completes and every logged metric is finite. LEARNING (reported,
  not gated): windowed deterministic-eval delta (mean of last third − first third
  of evals).

### Measured per-step FLORIS solve cost vs turbine count

Single-env, fixed layout, 270°/8 m/s (translated real layouts):

| farm        |  N | per-step FLORIS solve | env build | map (m)     |
| ----------- | -: | --------------------: | --------: | ----------- |
| Ablaincourt |  7 |                6.8 ms |    0.05 s | 2915 × 1533 |
| Ormonde     | 30 |               24.3 ms |    0.08 s | 5164 × 4978 |
| HornsRev1   | 80 |               73.8 ms |    0.32 s | 6512 × 4893 |

Roughly linear in turbine count (~0.9 ms/turbine over this range). Build is a
one-off; with a fixed layout resets do **not** rebuild the MDP, so per-step FLORIS
solve dominates collection.

## Results

**CAPABILITY PASS** — all four runs (2 farms × 2 variants) completed with finite
metrics. Real seed-0 runs, wandb **online** to project `wind-rl`
(deterministic-eval reward = total farm power, MW-scale).

**Primary — Ormonde (30 turbines), 40 iters:**

| variant           | first win | last win |   delta | collect s/iter | wall s/iter | wall total | wandb |
| ----------------- | --------: | -------: | ------: | -------------: | ----------: | ---------: | ----- |
| `set_transformer` |   45.4656 |  45.3516 | −0.1140 |          2.24  |       9.54  |    389.3 s | [0ujulabx](https://wandb.ai/mark-haoxiang/wind-rl/runs/0ujulabx) |
| `mlp`             |   46.9493 |  46.9386 | −0.0107 |          2.20  |       6.68  |    272.7 s | [bful8mzu](https://wandb.ai/mark-haoxiang/wind-rl/runs/bful8mzu) |

**Stretch — HornsRev1 (80 turbines), 5-iter feasibility smoke:**

| variant           | first win | last win |   delta | collect s/iter | wall s/iter | wall total | wandb |
| ----------------- | --------: | -------: | ------: | -------------: | ----------: | ---------: | ----- |
| `set_transformer` |   33.4203 |  33.4047 | −0.0156 |          6.36  |      24.36  |    129.8 s | [pgh9n94j](https://wandb.ai/mark-haoxiang/wind-rl/runs/pgh9n94j) |
| `mlp`             |   30.6043 |  31.7687 | +1.1644 |          6.26  |      16.70  |     90.4 s | [k14le0ow](https://wandb.ai/mark-haoxiang/wind-rl/runs/k14le0ow) |

### Parallel collection (first production use of `n_envs`)

`n_envs=null` auto-resolved to **20** workers at both scales (largest divisor of
`frames_per_batch=1000` that is `≤ min(1000//20, 20)`). Achieved collect time for
one 1000-frame batch, against the serial estimate `1000 × per-step`:

| farm      |  N | serial est. | 20-worker collect |  speedup |
| --------- | -: | ----------: | ----------------: | -------: |
| Ormonde   | 30 |      24.3 s |            ~2.2 s |    ~11×  |
| HornsRev1 | 80 |      73.8 s |            ~6.3 s |   ~11.7× |

The ~11–12× (not 20×) reflects fork/IPC overhead and reset-boundary jitter across
20 subprocess FLORIS workers; it is stable across a 2.7× jump in turbine count.
Peak RSS stayed well within the 60 GB box (20 × 80-turbine FLORIS instances + the
CPU trainer) — no memory pressure observed. Wall/iter is dominated at 80t by the
single-env eval + wake-field render (`set_transformer` 24.4 s/iter vs collect
6.4 s), not by collection.

### Learning outcome (honest)

- **Ormonde (30t): flat.** Both variants sit at their starting power and drift
  slightly down (`set_transformer` −0.11, `mlp` −0.01) over 40 iters. Ormonde is
  a 2-D cluster, not a wake-aligned row; under a single fixed 270° inflow only a
  minority of turbines sit in a strong wake, so the zero-yaw init is already near
  the achievable optimum and off-init exploration mostly costs a little power.
  Partial/flat learning at this scale-and-budget was anticipated; the capability
  claim (finite, stable training) is what this settles.
- **HornsRev1 (80t): `mlp` learns even in 5 iters** (30.60 → 31.77, **+1.16**),
  near-monotonically; `set_transformer` flat (33.40, and higher absolute power).
  More turbines → more downstream wake overlap → larger steering headroom, so a
  clear learning signal appears despite the tiny budget. This is the strongest
  learning signal of the four runs and a good sign for scaling the budget.

### Framework health

All four runs are PPO-stable and conservative, matching 0001/0003: approx-KL
≤ 0.0015, clip fraction ≤ 0.4 % every run — updates stay well inside the trust
region. Two honest, non-blocking caveats carried from 0001: pre-clip grad norm
sits ~8–14× above `max_grad_norm=1.0` (grad clipping active every step;
`set_transformer` runs hotter at ~13.7), and critic explained variance is ~0
(1e-5 to −0.009) — under fixed wind the episode return is nearly constant, so the
value head has almost no return variance to explain, yet normalised GAE
advantages still drive the policy (visibly so on HornsRev1 `mlp`). Each run logged
~34 metrics/iter plus a live wake-resolved eval-layout render and an interactive
HTML replay to wandb (media captured at 30t and 80t — the large-farm wake fields
render fine); final checkpoints (policy/critic state_dicts + config) are written
under `WIND_RL_WDIR/<experiment>_<variant>/` and the final one is uploaded as a
wandb Artifact.

## Decision

The co-design loop **trains on real farms**: it ingests a named wfcrl case,
translates its native (possibly negative) coordinate frame into an in-map scenario
without touching the physics, and runs MAPPO to completion with finite, healthy
telemetry at 30 and 80 turbines. The capability the paper framing needs is
demonstrated, and the parallel collector delivers a stable ~11–12× collect speedup
that makes 30–80-turbine training cheap (Ormonde 40 iters in ~4.5–6.5 min/variant;
HornsRev1 5 iters in ~1.5–2 min/variant). FLORIS per-step cost scales ~linearly
(~0.9 ms/turbine); at 80t the eval-render, not collection, is the per-iter cost
driver.

This run does **not** crown an architecture and does not claim strong learning at
30 turbines: under a single fixed inflow, Ormonde's cluster geometry leaves little
wake-steering headroom and both variants stay flat. The clearest next lever is
already visible — HornsRev1 `mlp`'s +1.16 in 5 iters says larger farms carry more
headroom and reward a longer budget. Next steps here: (1) a full HornsRev1 budget
(≥40 iters) to convert the smoke into a learning verdict; (2) varied / multi-
direction wind with a wind-conditioned policy and per-direction eval, where the
`set_transformer`'s wind-frame canonicalisation is expected to matter and the
fixed-wind flatness at Ormonde should lift.
