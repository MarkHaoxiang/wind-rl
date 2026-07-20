# 0004 — PPO tuning levers

## Hypothesis

0001's rich-telemetry runs on the fixed 3-turbine westerly row exposed a PPO
regime that *looked* stable but conservative:

1. **Grad clip saturates.** Pre-clip grad norm sat ~8x above
   `max_grad_norm = 1.0`, so the clip was active on essentially every optimiser
   step — the effective learning rate was ~8x below nominal. *Hypothesis:*
   raising `max_grad_norm` (letting more of the gradient through) speeds
   convergence without instability.
2. **Updates look deep inside the trust region.** 0001 reported clip fraction
   < 2 % and approx-KL < 0.005. *Hypothesis:* more aggressive updates (higher
   LR) are safe here.
3. **No exploration bonus** (`entropy_eps = 0`). Probably fine given the
   `TanhNormal` `initial_std = 0.3`, but one arm checks whether a small entropy
   bonus helps.

This experiment tests those levers with a small, hypothesis-driven grid (not a
blind search) and validates the best arm on a larger (6-turbine) row.

## Setup

- **Scenario (sweep).** Identical to 0001: 3 turbines in a westerly row at
  `[(252, 1000), (756, 1000), (1260, 1000)]`, map 2000x2000, `max_steps = 20`,
  wind **fixed** at 270 deg / 8 m/s (deterministic, high-headroom eval).
- **Architecture.** `mlp` — `MultiAgentMLP` actor/critic, shared params,
  `num_cells = 64`, `depth = 2`, `initial_std = 0.3` (the 0001 `mlp`).
- **Budget.** 40 iterations x 1000 frames (40k env steps) per run, **2 seeds**
  (0, 1) per arm, deterministic eval every iteration. Wall-clock ~215 s/run;
  sweep + validation ~62 min total.
- **Grid — a 2x3 factorial over `lr` x `max_grad_norm`, plus one entropy arm.**
  All other PPO settings are 0001's (clip 0.2, gamma 0.99, lambda 0.95,
  advantage normalised, Adam + cosine decay to `min_lr = 1e-4`, 8 epochs x 4
  minibatches):

  | arm                 | lr   | max_grad_norm | entropy_eps |
  | ------------------- | ---- | ------------- | ----------- |
  | `baseline` (0001)   | 3e-4 | 1.0           | 0.0         |
  | `gradnorm5`         | 3e-4 | 5.0           | 0.0         |
  | `gradnorm10`        | 3e-4 | 10.0          | 0.0         |
  | `lr1e-3`            | 1e-3 | 1.0           | 0.0         |
  | `gradnorm5_lr1e-3`  | 1e-3 | 5.0           | 0.0         |
  | `gradnorm10_lr1e-3` | 1e-3 | 10.0          | 0.0         |
  | `entropy1e-3`       | 3e-4 | 1.0           | 1e-3        |

- **Scoring.** Under a fixed budget every arm saturates to the same
  wake-steering optimum (~34.3 farm power), so the discriminating signal is
  *convergence speed*, scored by the **mean of the deterministic-eval trajectory
  (eval AUC)** — a higher mean means the arm climbed to the optimum sooner.
  Reported alongside: the 0001 windowed delta (mean of the last third of evals
  minus the first third, ±std over seeds), final windowed level, and
  final-iteration stability telemetry (clip fraction, approx-KL, pre-clip grad
  norm).
- **Verdict (asserted in `run.py`).** The sweep PASSes iff every run (sweep +
  validation) completes with all logged metrics finite; it exits nonzero
  otherwise. All runs logged **online** to wandb project `wind-rl`, grouped and
  tagged per arm.
- **Winner + generalisation check.** run.py picks the max-AUC arm subject to a
  stability gate (mean final approx-KL < 0.05) and re-runs it on a **6-turbine**
  westerly row (x = 252..2772 at 504 m, map 3200x2000, 20 iters, 2 seeds). The
  gate excluded every learning arm (see Results), so it fell back to `baseline`;
  `gradnorm5` (the raw max-AUC arm) was validated separately as a cross-check.

## Results

**SWEEP PASS** — all 18 runs (14 sweep + 4 validation) completed with finite
metrics. Real runs, seeds 0/1, wandb **online** to `wind-rl`.

### Sweep (3-turbine), 2 seeds/arm

| arm                 | delta (mean±std) | last-win | **eval AUC** | clip frac | final KL | grad norm | wandb (s0 / s1)   |
| ------------------- | ---------------: | -------: | -----------: | --------: | -------: | --------: | ----------------- |
| `baseline`          |  +1.915 ± 0.318  |  34.227  |   33.432     |   0.098   |  0.203   |   8.047   | [h37y1svi] / [rxugl6av] |
| `gradnorm5`         |  +1.861 ± 0.418  |  34.345  | **33.601**   |   0.109   |  0.206   |   8.056   | [hi3v3mm9] / [fk1jl8gh] |
| `gradnorm10`        |  +1.836 ± 0.435  |  34.309  |   33.580     |   0.107   |  0.218   |   8.061   | [zsxambnq] / [i6700q9v] |
| `lr1e-3`            |  +1.998 ± 0.581  |  34.329  |   33.510     |   0.110   |  0.293   |   8.063   | [xzfrir2g] / [7t33rn5b] |
| `gradnorm5_lr1e-3`  |  +1.709 ± 0.333  |  34.125  |   33.486     |   0.121   |  0.272   |   8.060   | [shibjwpb] / [i3p1ysw0] |
| `gradnorm10_lr1e-3` |  +1.627 ± 0.445  |  34.168  |   33.539     |   0.123   |  0.350   |   8.058   | [25e99f42] / [i5tdx74o] |
| `entropy1e-3`       |  +0.279 ± 0.473  |  32.808  |   32.795     |   0.059   |  0.131   |   8.044   | [pmle2o68] / [3jx0lkd3] |

Links: `https://wandb.ai/mark-haoxiang/wind-rl/runs/<id>`.

Three findings:

1. **Loosening the grad clip or raising LR does not speed convergence beyond
   seed noise.** The six no-entropy arms all land in a 0.17-wide AUC band
   (33.43–33.60), while the per-arm seed std is 0.3–0.6. The best AUC arm,
   `gradnorm5` (+0.17 over baseline), has a *lower* mean windowed delta than
   baseline (1.86 vs 1.92) — the ranking is inside the noise floor. The **seed**
   effect dwarfs the arm effect: every arm's seed-0 run starts its first window
   near 32.0 and seed-1 near 32.8, and this offset, not the PPO knob, drives the
   spread. Raising LR to 1e-3 buys no AUC and pushes final KL up (0.29–0.35 vs
   0.20); combining LR with a loose clip gives the highest KL (0.35) and the
   lowest deltas.

2. **The entropy bonus is actively harmful here.** `entropy1e-3` is the only arm
   clearly off the pack: AUC 32.80 (~0.8 below baseline), delta +0.28, and one
   seed even regressed (-0.19). The bonus keeps the policy stochastic, so it
   never tightens onto the deterministic wake-steering optimum that the eval
   rewards. This confirms 0001's call that `entropy_eps = 0` is correct in this
   deterministic, high-headroom regime.

3. **0001's "deep inside the trust region" reading did not hold at steady
   state.** Final-iteration approx-KL is **0.20–0.35** for every *learning* arm
   (not the ~0.003 0001 reported), and clip fraction is **~0.10** (not < 2 %).
   The policy is still taking sizeable, clip-bound steps at the end of the 40
   iterations — 0001's low numbers reflected an early/averaged snapshot, not the
   converged regime. Pre-clip grad norm stays pinned at ~8.05 across *all* arms
   regardless of `max_grad_norm`; raising the cap from 1→5→10 does let
   progressively more gradient through, but the downstream effect on the eval
   trajectory is negligible. On this task the clip is simply not the
   convergence bottleneck — the wake-steering optimum is reached comfortably
   within budget at every setting.

The in-code stability gate (`_KL_CAP = 0.05`) was calibrated against 0001's
reported 0.003 KL; because the true steady-state KL is ~0.2, the gate excluded
every learning arm and `run.py` fell back to `baseline` as the "winner". The
recommendation below rests on the **AUC ranking**, not the gate; the gate only
selects which arm auto-validates. The PASS/FAIL verdict (all-finite) is
independent of it and is sound.

### Validation (6-turbine row), 2 seeds each

| arm         | delta (mean±std) | last-win | wandb (s0 / s1)   |
| ----------- | ---------------: | -------: | ----------------- |
| `baseline`  |  +2.642 ± 0.003  |  31.365  | [y6hte3pl] / [gudi27jd] |
| `gradnorm5` |  +2.599 ± 0.075  |  31.341  | [hegnazcn] / [m2g94ezh] |

Both learn wake steering on the larger row (matching 0001's `mlp` scale probe,
+3.20) and land on top of each other — `baseline` +2.642 vs `gradnorm5` +2.599,
final level 31.37 vs 31.34. The marginal 3-turbine edge does not reappear at
6 turbines; if anything baseline is a hair ahead. The "no meaningful winner"
result is robust across scales.

## Decision

**Keep the current PPO defaults: `lr = 3e-4`, `max_grad_norm = 1.0`,
`entropy_eps = 0.0`.** No swept lever produced a convergence-speed improvement
beyond seed noise on the 3-turbine benchmark, and none separated from baseline
on the 6-turbine validation. The one clear, reproducible effect — adding an
entropy bonus — is *harmful* in this deterministic regime (AUC 32.80 vs 33.43),
so `entropy_eps = 0` is affirmatively correct, not merely tolerable.

**No config change is recommended.** (This is a recommendation only — the
trainer defaults and 0001's config are owned by other agents and were not
edited.)

Two secondary notes for whoever revisits this:

- If a future regime shows the grad clip *actually* binding as a bottleneck,
  `max_grad_norm = 5` is the safe first step: it was indistinguishable from
  baseline here (marginally higher AUC, no KL/stability cost), so it costs
  nothing to raise and lets the full ~8-norm gradient through.
- The premise behind hypothesis 2 was wrong at steady state: PPO here is *not*
  sitting deep in the trust region at convergence (final KL ~0.2, clip
  ~10 %). The apparent conservatism in 0001's telemetry was a snapshot
  artifact. Any future "the updates are too timid, push harder" instinct on
  this task should be checked against the *final-iteration* KL, not an early one
  — the headroom that motivated this sweep is not there.

Caveat carried forward (from 0001): the fixed-wind, high-headroom regime is what
gives a clean, deterministic learning signal at this scale. These PPO defaults
are validated for that regime; a wind-conditioned policy under wfcrl's random
wind is a separate, harder tuning question.
