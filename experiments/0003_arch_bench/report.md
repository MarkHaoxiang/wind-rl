# 0003 — Architecture benchmark suite

## Hypothesis

The three architectures in the `ModelConfig` union — `mlp`, `gcn` (dense-adjacency
GCN, research v0), `set_transformer` (permutation-equivariant set transformer with
wind-frame canonicalisation, research v1) — can be ranked by two fast proxy tasks
(`docs/research/2026-07-19-geometric-architectures.md` §5): a supervised
value-regression **critic proxy** and a fixed-budget MAPPO **policy proxy**. Each
architecture must be **functional** — its critic beats predict-the-mean (`EV > 0`)
and every PPO seed completes without NaN. The **decisive** profile additionally
tests whether, once the regime actually exercises geometric bias (varied wind,
16 turbines, a real training budget), the ranking separates from seed noise and
reorders the fixed-wind result.

## Setup

Two config profiles share one harness (`run.py`, tier loop over
`{scenario, variants}`; verdict asserted in code):

- **`config` (quick, default).** One fixed 8-turbine layout, **fixed** wind
  270°/8 m/s, 8 PPO iters × 3 seeds. Fast-iteration plumbing check
  (`WANDB_MODE=disabled`, ~12 min).
- **`decisive` (`--config-name decisive`).** The upgrade the first run's Decision
  called for. Two tiers — **8** and **16** turbines — each over all three archs.
  - **Varied wind.** `fixed_wind_direction` unset ⇒ wfcrl samples a random inflow
    every episode, so a wind-conditioned, rotation-aware policy must generalise
    across directions. Return variance exists again, so critic EV is meaningful
    (vs the fixed-wind regime where the only signal was the yaw trajectory). Eval
    averages **5** random-wind episodes/iter so the windowed delta tracks learning,
    not the wind draw.
  - **Budget.** Policy proxy **20 PPO iters × 3 seeds** {0,1,2} per (arch, tier),
    identical PPO hyperparameters across archs (clip 0.2, γ 0.99, λ 0.95, Adam
    3e-4 cosine→1e-4, 8 epochs × 4 minibatches). Critic proxy: 300 random-policy
    rollouts under varied wind (new cache key), targets standardized by train
    stats (predict-the-mean ≡ `MSE=1`/`EV=0`), identical optimiser budget across
    archs (Adam 2e-3, 2500 steps, batch 256, 80/20 split).
- **Device.** CPU (FLORIS is CPU-bound; a GPU is slower via host↔device transfer,
  and CPU is where the `set_transformer` attention vmap-fallback would show).
- **Score.** Critic: validation MSE + explained variance. Policy: windowed
  deterministic-eval reward delta (mean of last third − first third of evals) per
  seed, mean ± std across seeds, plus wall-clock/iter. Layouts, checkpoints and the
  varied-wind dataset cache land under `WIND_RL_WDIR`; policy runs log online to
  wandb project `wind-rl`, grouped per arch-tier (3 seeds/group), tagged
  `{arch, tier, sN, decisive}`.

**Verdict (asserted in `run.py`).** Every architecture, every tier must be
functional — critic `EV > 0` **and** every PPO seed finite. Exits nonzero iff any
cell is non-functional. It does **not** assert a ranking.

## Results — decisive profile

**BENCHMARK PASS** — every architecture is functional at both tiers. Real run,
CPU, wandb online. Total wall-clock **≈ 90 min** (t8 ≈ 38 min, t16 ≈ 52 min).

**8 turbines, varied wind**

| arch              | critic EV | critic MSE | Δ mean ± std | per-seed Δ (s0/s1/s2)    | s/iter | params  |
| ----------------- | --------: | ---------: | -----------: | ----------------------- | -----: | ------: |
| `mlp`             |  +0.5713  |    0.4282  | +0.174 ±0.99 | +1.566 / −0.479 / −0.566 |  11.61 |  11 459 |
| `gcn`             |  +0.1784  |    0.8092  | +0.554 ±0.32 | +0.218 / +0.461 / +0.983 |  11.31 |   9 219 |
| `set_transformer` |  +0.7854  |    0.2123  | +0.443 ±0.89 | −0.325 / +1.694 / −0.039 |  11.73 | 135 043 |

**16 turbines, varied wind**

| arch              | critic EV | critic MSE | Δ mean ± std | per-seed Δ (s0/s1/s2)    | s/iter | params  |
| ----------------- | --------: | ---------: | -----------: | ----------------------- | -----: | ------: |
| `mlp`             |  +0.7482  |    0.2526  | +0.345 ±0.98 | −1.008 / +1.293 / +0.750 |  18.72 |  14 019 |
| `gcn`             |  +0.2855  |    0.7001  | +0.581 ±0.25 | +0.329 / +0.494 / +0.919 |  16.51 |   9 219 |
| `set_transformer` |  +0.8800  |    0.1196  | −0.228 ±0.66 | −0.015 / −1.114 / +0.446 |  17.13 | 135 043 |

- **Critic proxy — decisive and stable.** Ordering is `set_transformer` >
  `mlp` > `gcn` at **both** scales, and every architecture *improves* with more
  turbines (more inflow geometry to fit): `set_transformer` 0.785→0.880, `mlp`
  0.571→0.748, `gcn` 0.178→0.286. `set_transformer` fits the varied-wind value
  function best by a clear margin; `gcn`'s single mean-pooled scalar value
  bottlenecks it.
- **Policy proxy — one architecture separates from noise: `gcn`.** It is the only
  architecture with **all six** seed deltas positive, the tightest variance
  (std 0.32 / 0.25 vs 0.66–0.99 for the others), and the highest mean at **both**
  tiers (+0.554, +0.581). `mlp` and `set_transformer` straddle zero (deltas flip
  sign across seeds, std ~0.7–1.0); `set_transformer`'s 16t mean is negative but
  within its own noise band. This is a **reorder** of the fixed-wind run, where
  `gcn` was the clear third — varied wind rewards its locality bias, exactly the
  lever the first Decision predicted might move `gcn` vs `mlp`.
- **Cost / Pareto.** Env-dominated: at 16t all three land within ~2 s/iter, and
  `gcn` is *cheapest* (16.51 s) — the `set_transformer` vmap-fallback tax stays
  invisible behind FLORIS. `gcn`/`set_transformer` are parameter-count invariant
  in `N` (9 219 / 135 043 at both tiers, permutation-equivariant); `mlp`'s
  centralized critic grows with `N` (11 459→14 019).

## Decision

The critic proxy and the policy proxy point in **different directions**, and that
split is the finding:

- **Value-fitting capacity → `set_transformer`, decisively.** Best critic EV at
  both scales (0.785, 0.880) and the only architecture whose lead *widens* with
  turbines. But this capacity does **not** convert to policy gains at a 20-iter
  budget — its policy delta straddles zero at 8t and is negative-in-noise at 16t.
- **Reliable policy improvement → `gcn`, decisively.** The only architecture with
  uniformly positive, low-variance deltas across all six (tier × seed) cells, best
  mean at both scales, lowest params, cheapest 16t wall-clock. Its weak critic is
  a genuine caveat — a poor value function should hurt PPO — but its strong
  locality prior evidently regularises the policy into consistent, if modest, gains.
- **`mlp` is dominated** on the Pareto front: a middle critic that must grow with
  `N`, and the highest-variance policy (straddles zero at both scales).

**Promotion recommendation (confidence: MODERATE).** For the main MAPPO policy
pipeline at the current operating point (8–16 turbines, varied wind, short
budget), promote **`gcn`** as the *reliability* pick — it is the one architecture
whose policy improvement is already separated from seed noise, at the lowest cost.
Carry **`set_transformer`** as the *capacity* front-runner to revisit at a longer
training budget, where its decisive, scale-improving critic-EV lead has room to
translate into policy gains and its short-budget policy regression can be retested.
Deprioritise **`mlp`**.

Confidence is moderate, not high, because only the `gcn` policy signal has cleared
seed noise; `mlp` vs `set_transformer` on the policy proxy remains inconclusive
(deltas within their own std). What would settle `gcn` vs `set_transformer`
definitively: run both to **convergence** (not a 20-iter probe) at 8 and 16
turbines under varied wind — the critic proxy says `set_transformer` has the higher
ceiling, so the open question is purely whether more optimisation lets its policy
realise it before `gcn`'s reliable-but-modest curve plateaus.
