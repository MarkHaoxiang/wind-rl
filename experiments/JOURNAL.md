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
