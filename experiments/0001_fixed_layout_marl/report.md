# 0001 — Fixed-layout MARL benchmark (unified framework)

## Framework

One framework that trains a list of **config variants** under an identical PPO
budget on a fixed wind-farm layout, harvests a comparable per-run metric, prints a
cross-variant table, and asserts a per-variant verdict in code. The shape is shared
across three concerns that used to be separate experiments; each is now a config
entry point selected with `config=<name>` (default `config`):

| entry point          | variants differ by      | gate       | layout                         |
| -------------------- | ----------------------- | ---------- | ------------------------------ |
| `config` (default)   | architecture (`model`)  | `improves` | fixed 3-turbine westerly row   |
| `config=ppo_sweep`   | PPO levers (`ppo`)      | `finite`   | fixed 3-turbine westerly row   |
| `config=real_farms`  | architecture (`model`)  | `finite`   | named wfcrl farm (translated)  |

Run with wandb online (default) or disabled for a plumbing check:

```bash
uv run python experiments/0001_fixed_layout_marl/run.py                 # arch benchmark
uv run python experiments/0001_fixed_layout_marl/run.py config=ppo_sweep
uv run python experiments/0001_fixed_layout_marl/run.py config=real_farms farm.name=HornsRev1
WIND_RL_WANDB_MODE=disabled uv run python experiments/0001_fixed_layout_marl/run.py base.n_iters=2
```

### Machinery (`wind_rl.experiment`)

`run.py` is thin (compose config → `run_sweep` → `summarize`/`format_table` →
per-variant verdict → exit code); everything reusable lives in the library:

- **`sweep.py`** — `Variant` (name + `TrainingConfig` overrides, or a full config)
  and `run_sweep(base, variants, seeds)`: the per-`(variant, seed)` loop that builds
  the `TrainingConfig`, runs `MappoTrainer`, times it, and reduces its history to a
  typed `RunResult` (windowed first/last/delta, eval AUC, wall-clock, finiteness).
  It sets `WANDB_RUN_GROUP` / `WANDB_TAGS` per run so a variant's seeds share a wandb
  group, and calls `wandb.teardown()` between runs so those env vars are re-read
  rather than cached from the first init.
- **`table.py`** — aggregates a variant's seeds to mean ± population-std and renders
  the comparison table.
- **`verdict.py`** — `windowed_delta` (first-third vs last-third mean of the
  deterministic-eval trajectory) and the parameterised gates `improves(margin)`
  (learning: last window beats own first window) and `is_finite()` (capability: all
  metrics finite). No experiment-specific threshold is baked in.
- Real farms are resolved by `wind_rl.scenario.resolve_real_farm`: it fetches the
  named wfcrl layout and translates its native (possibly negative) metre frame so
  the bounding-box corner sits at `(margin, margin)`, deriving the map bounds from
  `bbox + margin`. Wake physics depend only on relative positions, so this keeps
  every coordinate positive and in-map (what the mlp position normalisation, the
  renderer axes, and the `layout` observation Box all assume) without altering the
  physics. The scenario template (`base.scenario`) supplies max_steps, spacing, and
  fixed wind; only name and geometry are derived.

## Results (default arch benchmark)

The default `config` benchmarks architectures on the fixed 3-turbine westerly row
`[(252,1000),(756,1000),(1260,1000)]` (map 2000×2000, `max_steps=20`), wind fixed at
270°/8 m/s (deterministic eval, large wake-steering headroom), 40 iters × 1000
frames, `gate=improves`. `mlp` and `gcn` learn wake steering (real seed-0 runs, wandb
online to `wind-rl`; reward = total farm power):

| variant | first window | last window |   delta | wall-clock |
| ------- | -----------: | ----------: | ------: | ---------: |
| `mlp`   |      32.2381 |     33.5124 | +1.2744 |    219.5 s |
| `gcn`   |      32.3501 |     33.2348 | +0.8847 |    274.8 s |

Both trajectories dip in the first few iterations (exploration off the zero-yaw
init) then climb near-monotonically. Reference probes: zero-yaw ≈ 30.6/episode, best
fixed upstream steering (+25°) ≈ 35.6 — both learned policies recover most of the
available gain. PPO is stable and conservative (clip fraction < 2 %, approx-KL <
0.005 at this snapshot); two honest, non-blocking caveats the telemetry exposes:
pre-clip grad norm sits ~8× above `max_grad_norm=1.0` (clip active every step), and
critic explained variance is ~0 (under fixed wind the return is nearly constant, yet
normalised GAE advantages still drive the policy). `set_transformer` (0003's
front-runner) is included in the default variant set and is known to train on this
regime (0003, 0005), but has not been run at full arch-benchmark budget here.

The benchmark does **not** crown a winner at smoke scale: the KNN graph collapses to
a nearly fully-connected 3-node graph and the headroom is small, so the `mlp`/`gcn`
gap is not a meaningful ranking. Separating architectures needs a larger budget and
more turbines (the `ppo_sweep` and `real_farms` entry points; 0003's proxy suite).

## Decision

The fixed-layout MARL benchmark is settled as a **single framework** with three
config entry points sharing one library loop, table, and gate set. The default arch
benchmark PASSes (both `mlp` and `gcn` learn wake steering). The PPO-lever sweep and
the real-farm capability study are the same framework under different configs and
gates; their concluded verdicts are recorded in `experiments/JOURNAL.md` (no lever
beat the PPO defaults beyond seed noise; MAPPO trains to completion with finite,
stable telemetry on real farms up to 80 turbines with a ~11–12× parallel-collection
speedup). New studies of this shape are new **config variant sets** here, not new
experiment directories.

Caveat carried forward: the fixed-wind regime is what exposes a clean learning
signal at this scale. Under wfcrl's random wind a wind-conditioned policy is required
and the average headroom is small — a stronger test for later, ideally with a
per-direction evaluation protocol.
