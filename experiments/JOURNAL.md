# Journal

Append-only. One line per concluded finding: `NNNN_slug` — verdict — pointer
to the run/report. See `experiments/README.md` for the contract.

- `0001_mappo_smoke` — PASS — MAPPO walking skeleton learns wake steering on a
  fixed-wind 3-turbine FLORIS row: deterministic eval 32.05 -> 33.98, windowed
  +1.27 (208 s). See `0001_mappo_smoke/report.md`.
- `0002_flowmap_prior` — PASS — pure-FM flow-map prior (mfm consistency loss) on
  the 3-turbine procedural distribution: 4-step samples reach raw feasibility
  0.576 and projected (SLSQP) feasibility 1.000 >= 0.95 (NFE=4, 2.9 s train).
  See `0002_flowmap_prior/report.md`.
- `0001_fixed_layout_marl` — PASS — fixed-layout MARL benchmark: both `mlp` and
  `gcn` learn wake steering on the fixed-wind 3-turbine row under an identical
  PPO budget (windowed eval mlp 32.24 -> 33.51 +1.27, gcn 32.35 -> 33.23 +0.88;
  ~7 min total). No winner crowned at smoke scale. See
  `0001_fixed_layout_marl/report.md`.
- `0001_fixed_layout_marl` (rich-telemetry rerun) — PASS — reproduces the
  benchmark online to wandb `wind-rl` with ~34 metrics/iter; telemetry is
  healthy (clip <2%, KL <0.005) with two honest flags (grad norm ~8x the clip
  budget; critic explained-variance ~0 under fixed wind). 6-turbine `mlp` scale
  probe also learns (28.82 -> 32.02 +3.20). See `0001_fixed_layout_marl/report.md`.
- `0003_arch_bench` — PASS — architecture-benchmark suite ranks `mlp`/`gcn`/
  `set_transformer` on a fixed 8-turbine layout via two fast proxies under
  identical budgets: critic value-regression EV (set_transformer 0.685 > mlp
  0.592 > gcn 0.185) and 8-iter/3-seed MAPPO delta (only set_transformer mean
  +0.158). All three FUNCTIONAL (EV>0, no NaN); no winner crowned — deltas are
  within seed noise and fixed wind under-exercises geometric bias (~12 min). See
  `0003_arch_bench/report.md`.
- `0003_arch_bench` (decisive: varied wind, 8+16-turbine tiers, 20 iters × 3
  seeds) — PASS — all archs FUNCTIONAL at both scales, and the proxies split:
  critic-EV decisively `set_transformer` > `mlp` > `gcn` at both tiers and
  scale-improving (0.79→0.88 / 0.57→0.75 / 0.18→0.29), but the policy proxy
  reorders — `gcn` is the only arch with all-positive, low-variance deltas
  (+0.55±0.32 @8t, +0.58±0.25 @16t) while `mlp`/`set_transformer` straddle zero.
  Recommend `gcn` (reliability, moderate confidence) now, `set_transformer`
  (capacity) at longer budgets; `mlp` dominated (~90 min, wandb online). See
  `0003_arch_bench/report.md`.
- `0004_ppo_tuning` — PASS — PPO tuning-lever sweep (2x3 factorial over lr x
  max_grad_norm + entropy, `mlp`, 3-turbine, 2 seeds/arm, online). No lever beats
  baseline beyond seed noise: eval-AUC band 33.43–33.60 vs per-arm std 0.3–0.6;
  entropy_eps=1e-3 is harmful (AUC 32.80). Steady-state KL is ~0.2 / clip ~10%
  (not 0001's snapshot 0.003/<2%), so the clip is not the bottleneck. 6-turbine
  validation ties baseline (+2.64) and gradnorm5 (+2.60). Decision: keep
  lr=3e-4, max_grad_norm=1.0, entropy_eps=0. See `0004_ppo_tuning/report.md`.
- `0005_real_farm` — PASS (capability) — first MARL training on REAL wfcrl
  layouts: Ormonde (30t, 40 iters) and HornsRev1 (80t, 5-iter smoke), both
  variants, online. Runs complete with finite, PPO-stable telemetry at
  production scale (real coords translated in-map, physics preserved). First
  production use of the parallel collector: `n_envs` auto→20 gives ~11–12×
  collect speedup (FLORIS ~0.9 ms/turbine/step). Learning honest: Ormonde flat
  (fixed-wind cluster, little headroom); HornsRev1 `mlp` learns +1.16 in 5 iters.
  See `0005_real_farm/report.md`.
