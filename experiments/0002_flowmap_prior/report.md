# 0002 — Flow-map layout prior (unconditional, pure FM)

## Hypothesis

A flow-map layout generator (mfm's SI consistency loss in the pure
flow-matching regime — diagonal `s == u` only, off-diagonal consistency term
parked) trained on the 3-turbine procedural feasible distribution learns that
distribution well enough that **few-step (4-step Euler) unconditional samples,
after hard SLSQP feasibility projection, are essentially all feasible**, while
raw pre-projection feasibility clears a modest floor. This validates the T7
prior + `FlowMapDesigner` path before any critic guidance is added.

## Setup

- **Scenario.** 3 turbines, map 2000x2000, `min_distance = 400`. Training data
  are procedurally sampled feasible layouts (`sample_feasible_layout`,
  min-distance rejection) — the same feasible set the verdict scores against.
- **Model.** `FlowMapModel`: time-conditioned MLP velocity field over the
  flattened `2N = 6` layout vector, coordinates axis-normalised to `[-1, 1]`.
  width 128, depth 4, sinusoidal time embedding (128). Conforms to mfm's
  `BaseModel.v(s, u, x, t_cond, x_cond, ...)` so mfm's loss drives it unchanged.
- **Loss.** `mfm.losses.get_consistency_loss_fn` with an attribute-bag config in
  the pure-FM regime: fully unconditional (`t_cond == 0` always), `l2` weighting,
  `num_warmup_steps` parked at 1e12 so the off-diagonal `s < u` term never turns
  on. Reduces to plain velocity regression on `x1 - x0`.
- **Training.** 4096 procedural layouts, 3000 Adam iters, batch 256, lr 1e-3,
  seed 0, CPU.
- **Sampler.** Few-step Euler integration of the probability-flow ODE using the
  trained instantaneous velocity `v(t, t, x)`; 4 steps (NFE = 4). Samples are
  denormalised to map metres and projected with hard SLSQP (Euclidean projection
  onto in-bounds + pairwise `>= min_distance`, with a 0.1% margin so the closed
  constraint is cleared).
- **Verdict (asserted in `run.py`).** PASS iff projected feasibility of 512
  samples `>= 0.95` **and** raw (pre-projection) feasibility `>= 0.30` (floor).

## Results

- **PASS.** Wall-clock **2.9 s** train (3000 iters), well under the "minutes"
  budget.
- Loss (first 10% window -> last 10% window): **6.42 -> 5.41**. The floor is the
  irreducible FM velocity-regression variance (target `x1 - x0` with random
  `x0`), not a convergence failure.
- **Raw 4-step feasibility: 0.576** (above the 0.30 floor). A crude 4-step Euler
  sampler on a pure-FM prior lands ~58% of layouts already feasible; the rest
  violate min-distance by small margins.
- **Projected (SLSQP) feasibility: 1.0000** (>= 0.95 threshold). Every sample is
  a small Euclidean nudge away from feasibility.
- Checkpoint written to `WIND_RL_WDIR/0002_flowmap_prior/prior.pt`; the
  `FlowMapDesigner` loads it, samples 4-step, projects, and drives feasible,
  varying env resets (see `tests/generative/test_flowmap_designer_sim.py`).

## Decision

The T7 flow-map prior + `FlowMapDesigner` path works end to end: mfm's
consistency loss trains a usable unconditional layout prior in seconds, few-step
sampling plus hard projection yields fully feasible layouts, and NFE (= 4) is
logged for the C1 efficiency claim. This is the v0 baseline the architecture
benchmark suite (owner decision 3) and critic guidance (deferred with
`ValueLearner`) build on.

Caveats for later work: (1) raw feasibility 0.58 at 4 steps reflects the pure-FM
regime — turning on mfm's off-diagonal consistency term (a true flow *map*)
should lift raw few-step feasibility and is the natural next iteration. (2) The
MLP-on-flattened-coords generator is deliberately the simplest v0; the
permutation-invariant / equivariant architectures (T5/T8) are subordinate to
what the benchmark suite concludes. (3) The verdict scores feasibility against
the *same* procedural distribution the prior trained on; power-capture quality
under co-design is a separate, later claim.
