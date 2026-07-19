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
