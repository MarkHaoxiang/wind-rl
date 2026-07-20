# 0003 — Architecture benchmark suite

## Hypothesis

The three architectures in the `ModelConfig` union — `mlp`, `gcn` (dense-adjacency
GCN, research v0), `set_transformer` (permutation-equivariant set transformer with
wind-frame canonicalisation, research v1) — can be ranked *cheaply*, in minutes
rather than full training runs, by two fast proxy tasks
(`docs/research/2026-07-19-geometric-architectures.md` §5): a supervised
value-regression **critic proxy** and a fixed-budget MAPPO **policy proxy**. Each
architecture is expected to be **functional** — its critic beats a
predict-the-mean baseline (`EV > 0`) and its PPO runs complete without NaN. The
suite is designed to *rank*, not to crown a winner at this scale.

## Setup

- **Fixed layout.** One feasible 8-turbine layout, sampled once via
  `design.geometry.sample_feasible_layout(scenario, rng(layout_seed=0))` and held
  constant. Map 2000×2000, min spacing 400 m. Wind is **fixed** at 270°/8 m/s, so
  the only signal in the critic proxy is the yaw trajectory (positions and wind
  are constant across every sample) — a clean, geometry-light regression that
  isolates critic fitting capacity.
- **Device.** Pinned to **CPU**. FLORIS is CPU-bound, so a GPU does not help
  (measured: the same run is *slower* on CUDA due to host↔device transfer around
  the CPU env), and CPU is where the `set_transformer` attention vmap-fallback
  cost is honestly visible.

**Critic proxy — supervised value regression.** A dataset of per-agent
observations + empirical discounted returns (Monte-Carlo within episode,
γ=0.99) is collected from **300 random-policy rollouts** (20 steps × 8 agents →
6000 timestep samples) and cached under `WIND_RL_WDIR/0003_arch_bench/`. Targets
are standardized by train statistics, so predict-the-mean is exactly `MSE=1 /
EV=0`. Each architecture's critic — built through the shared
`build_actor_critic` union entry — is trained under an **identical budget**
(Adam, lr 2e-3, 2500 steps, batch 256, 80/20 split) and scored by validation MSE
and explained variance. `EV > 0` is the functional gate. The budget is sized so
the slowest-converging critic (the MLP is a slow starter) clears the gate with
margin; it is *not* tuned to favour any architecture.

**Policy proxy — fixed-budget MAPPO.** Short identical-budget runs via the shared
`MappoTrainer`: **8 iterations × 1000 frames**, **3 seeds** {0,1,2}, PPO clip 0.2,
γ 0.99, λ 0.95, Adam 3e-4 (cosine → 1e-4), 8 epochs × 4 minibatches. Score: the
windowed deterministic-eval reward delta (mean of the last third of evals minus
the first third) per seed and averaged, plus wall-clock per iteration. Functional
iff every seed completes with all-finite metrics.

**Verdict (asserted in `run.py`).** Every architecture must be functional — critic
`EV > ev_gate` (0.0) **and** every PPO seed finite. The run exits nonzero iff any
architecture is non-functional. It does **not** assert a ranking.

## Results

**BENCHMARK PASS** — every architecture is functional. Real run, CPU, seed 0,
`WANDB_MODE=disabled`. Total wall-clock **≈ 12 min** (critic proxy ≈ 90 s incl.
dataset generation; policy proxy 3 × ~200 s), well under the ~40 min budget.

| arch              | critic EV | critic MSE | Δ(s0)  | Δ(s1)  | Δ(s2)  | Δ mean  | s/iter | params  | verdict    |
| ----------------- | --------: | ---------: | -----: | -----: | -----: | ------: | -----: | ------: | ---------- |
| `mlp`             |  +0.5920  |    0.4108  | +0.356 | −0.423 | −0.107 | −0.058  |  8.29  |  11 459 | FUNCTIONAL |
| `gcn`             |  +0.1852  |    0.8047  | +0.106 | −0.238 | −0.372 | −0.168  |  8.31  |   9 219 | FUNCTIONAL |
| `set_transformer` |  +0.6854  |    0.3162  | +0.115 | +0.018 | +0.342 | +0.158  |  8.72  | 135 043 | FUNCTIONAL |

- **Critic proxy (the discriminating signal).** `set_transformer` (EV 0.685) and
  `mlp` (EV 0.592) both fit the value function well; `gcn` (EV 0.185) captures far
  less variance. The GCN's constraints — KNN dense adjacency, `tanh`
  message-passing, and a *single* mean-pooled scalar value broadcast to all agents
  — bottleneck its capacity on this per-agent-yaw regression, whereas the MLP's
  centralized critic and the transformer's per-token pooling retain more of the
  signal.
- **Policy proxy.** At an 8-iteration budget the windowed deltas are small and
  seed-dependent (both signs), on a reward scale of ~52–58. `set_transformer` is
  the only architecture with a positive mean delta (+0.158) *and* consistent sign
  behaviour across seeds; `mlp` and `gcn` straddle zero. This is a learning-signal
  probe, not convergence — the functional gate (no NaN, run completes) is what all
  three clear.
- **Wall-clock / vmap-fallback.** Per-iteration cost is dominated by FLORIS env
  collection (~8 s), so all three land within ~5% of each other. The
  `set_transformer`'s known CPU SDPA vmap-fallback (`nn.MultiheadAttention`'s math
  path, taken because the fused path lacks a vmap batching rule under GAE/PPO) is
  real — measured ~4× in the *update* step alone (≈0.24 s vs ≈0.06 s) — but swamped
  by env time here, showing up only as the +0.4 s/iter gap.
- **Params.** `set_transformer` (135 k) is >10× the `mlp` (11.5 k) and `gcn`
  (9.2 k).

## Decision

The suite is settled as a **framework**: two fast proxies, identical budgets, a
scripted functional gate, and one comparison table — extended by appending a
`variant`. All three architectures PASS.

**What the numbers suggest re promotion.** On the cleaner of the two signals (the
critic proxy) `set_transformer` leads (EV 0.685), edging the strong-and-cheap
`mlp` baseline (0.592), with `gcn` a clear third (0.185); `set_transformer` is
also the only architecture with a positive mean policy delta. Taken at face value
this points toward **`set_transformer`** for promotion, with **`mlp`** as the
value-for-compute floor (most of the EV story at <1/10 the parameters and no
vmap-fallback tax) and **`gcn`** the weakest fit. But this is a *suggestion, not a
verdict* — deliberately, per the suite's design.

**Why it is not decisive, and what a decisive comparison needs.**
1. **Policy deltas are within seed noise.** 8 iterations is far too short; the
   signed deltas flip across seeds. A decisive policy comparison needs
   substantially more iterations and more seeds to separate learning from rollout
   variance.
2. **The regime under-exercises geometric bias.** Fixed wind + fixed layout means
   the transformer's permutation-equivariance / canonicalisation and the GCN's
   locality do little work — exactly the levers those architectures exist for.
   Scaling to more turbines and to *varied* wind (where a wind-conditioned,
   rotation-aware policy must generalise across inflow directions) is where a real
   ranking would emerge, and could plausibly reorder `gcn` vs `mlp`.
3. **Cost is not yet on the scale.** The vmap-fallback tax is invisible behind
   FLORIS here; on a cheaper/faster env or GPU-resident rollout it would matter,
   and the quality-vs-wall-clock Pareto (not quality alone) is the promotion
   criterion the research doc mandates.

So: promote nothing on this run. The table is the deliverable; `set_transformer`
is the front-runner to *carry into* a larger-budget, more-turbines, varied-wind
comparison, with `mlp` as the baseline it must beat on the Pareto front.
