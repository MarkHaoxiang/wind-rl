# wind-rl — Architecture & Research Plan

Date: 2026-07-23 · Author: planning agent · Status: DRAFT (owner review required)

wind-rl is the applications-paper successor to the WFCRL section of **DiCoDe**
(Li, Amir, Prorok, "Scaling Multi-Agent Environment Co-Design with Diffusion
Models", arXiv:2511.03100, ICML 2026). DiCoDe validated diffusion-guided
wind-farm layout co-design only to 16 turbines on FLORIS with yaw-only control.
wind-rl scales this substantially using **permutation/E(n)-equivariant
architectures** and **domain-specific fine-tuning**, and pushes to real
32/64/92-turbine farms.

---

## 1. Research framing

### Thesis
Diffusion-guided environment co-design does not scale (fragile guidance
annealing, MLP-on-flattened-coords generator with no geometric structure,
REINFORCE baseline collapses past 4 turbines). wind-rl shows that an
**equivariant layout prior + equivariant MAPPO policy** co-designed jointly
scales to full real-world wind farms, and that fine-tuning a generic pretrained
layout prior to a specific site's boundary + wind rose yields site-optimised
layouts.

### Claims the paper would make
- **C1 (scale).** Co-design remains stable and beats all non-generative
  baselines at 32/64/92 turbines, where DiCoDe was never evaluated and the
  REINFORCE RL baseline collapses.
- **C2 (architecture).** Permutation-invariant / E(n)-equivariant layout
  generator and policy improve sample efficiency and final power vs the
  MLP/hand-engineered-wind-GNN of DiCoDe at matched compute; equivariance
  removes the need for the fragile guidance-weight annealing schedule.
- **C3 (fine-tuning).** A layout prior pretrained on generic procedural layouts,
  then fine-tuned per site (boundary polygon + measured wind rose), produces
  higher-power constraint-satisfying layouts than a from-scratch or generic
  prior. (Secondary: FLORIS-pretrained policy transfers to FastFarm with
  fine-tuning at a fraction of from-scratch cost.)

### Baselines (all from DiCoDe's designer zoo — natural, already-implemented-in-spirit)
Random, Fixed, ManualCases (real published layouts), SamplingDesigner
(best-of-n by critic), GradientDescentDesigner (Adam ascent through critic),
ReinforceDesigner (per-turbine Gaussian REINFORCE — the "RL" baseline),
ReplayDesigner (prioritised population + mutation), and **DicodeDesigner**
(guided DDIM, PUG) as the headline prior-art competitor. wind-rl's contribution
— equivariant architectures, real-farm scale, and per-site fine-tuning — is
measured against these.

### "Domain-specific fine-tuned training" — chosen interpretation (DECISION, confirm)
Primary: **layout-prior fine-tuning.** Pretrain an unconditional layout
generator on procedurally generated feasible layouts (rejection / projected-
gradient sampling, as in DiCoDe `setup_scenario.py`), then fine-tune per real
site on that site's *boundary polygon constraints and measured wind-rose
distribution*, so the prior concentrates mass on high-power feasible regions for
that farm. Secondary track: **policy fidelity transfer** — MAPPO policy
pretrained on cheap FLORIS, fine-tuned on high-fidelity FastFarm. We propose to
pursue the layout-prior interpretation as the headline (it is what "fine-tuned"
most naturally means for a generative prior) and treat FastFarm transfer as a
stretch experiment. **Owner: confirm which is the paper's C3.**

### Milestone roadmap
- **M1 — Reproduce baseline.** MAPPO (Mava) on `windrl-engine` at small scale
  (2–3 then 8–16 turbines, yaw). Fidelity anchored by the FLORIS golden suite;
  match DiCoDe's power-capture numbers. Walking skeleton + infra (CI, tests,
  config, logging).
- **M2 — Architecture upgrades.** Permutation-invariant / E(n)-equivariant
  layout generator (real E-GNN, not DiCoDe's MLP stub) and equivariant MAPPO
  policy/critic. Demonstrate C2; drop guidance annealing.
- **M3 — Scale + fine-tune.** 32/64/92 turbines and real farm layouts
  (HornsRev1/2, Ormonde, WMR from `wfcrl.environments.data_cases`); per-site
  fine-tuning (C3); optional FastFarm transfer.
- **M4 — Paper experiments.** Full sweep across designers x scenarios x seeds,
  wall-clock/NFE benchmarks, ablations, figures. Evidence-gated (asserted
  thresholds, never eyeballed).

---

## 2. Code architecture

The stack is JAX-native (owner decision 2026-07-22). Simulation, training, and
the generative/design layer are separate packages in the **uv workspace**;
`experiments/` holds numbered frameworks (`NNNN_slug/run.py` + `conf/`,
`report.md`, owner-managed `JOURNAL.md`).

### Packages (one-line responsibilities)

```
packages/windrl-engine/   JAX wind-farm simulator — the environment.
                          Layers farm → physics → env (+ analysis side-car);
                          full GCH physics, FLORIS-faithful to machine
                          precision via committed goldens; batched envs ×
                          wind conditions; fidelity flag ("floris" |
                          "corrected") and selectable turbine library
                          (nrel5mw_v3 / v4). Source of truth:
                          docs/plans/2026-07-22-windrl-engine-design.md and
                          docs/plans/2026-07-22-jax-windfarm-step-spec.md.
packages/windrl-train/    MARL training on the engine via Mava MAPPO
                          (continuous actions, one agent per turbine).
                          Workspace-EXCLUDED: Mava pins jax==0.5.3 and
                          py<3.13, so this package owns a py3.12 venv and
                          lock; the jumanji-style wrapper adapts the engine's
                          single-farm functional core (Mava vmaps envs
                          itself). Zero Mava source edits.
packages/wind-rl/         Generative/co-design layer (torch, cu130 index) +
                          experiment harness:
  config.py               #   pydantic v2 Config base ("pydra" pattern).
  scenario.py             #   ScenarioConfig (procedural scenarios; real
                          #   farm layouts now live in windrl_engine.farm).
  generative/             #   DDIM layout generator, PUG guidance,
                          #   feasibility constraints (SLSQP projection).
  design/geometry.py      #   layout geometry utilities.
  experiment/             #   settings (WIND_RL_* env vars), wandb harness,
                          #   sweep/table/verdict machinery, cli.
```

### Reference fidelity (replaces the wfcrl dependency)

`packages/windrl-engine/tests/goldens/` freezes the reference:
`floris_v3.5.npz` (solver fields, verified 1e-12 against the live stack
before wfcrl removal), `wfcrl_env_trajectories.npz` (env behavior incl.
duty-cycle firing and the truncation boundary), `floris_v4.6.6.npz`
(latest-FLORIS turbine library). Reference tests assert against goldens, run
unfiltered on CI, and need no wfcrl/floris install; regeneration scripts run
FLORIS in isolated envs (`uv run --isolated --with floris==…`).

### Key interfaces

Engine: `solve_farm(layout, wind, yaw, …) -> FlowSolution`,
`turbine_powers`, functional `reset`/`step`, `BatchedWindFarmEnv`,
`analysis.{power_surface, aep, flow slices}` — signatures fixed in the engine
design doc. Training: `windrl_train.env.WindFarm` (jumanji contract) +
`windrl_train.train` (Mava ff_mappo entrypoint, hydra-composed). The
co-design `Designer` layer was deleted with the torch stack and will be
reintroduced JAX-side when co-design work starts (see T4).

### What to keep vs fix (from DiCoDe)
Kept in spirit, JAX-side: batched env evaluation, two-stage config with a
scenario registry, wandb logging (via Mava), no hardcoded paths
(`pydantic-settings`, `WIND_RL_WDIR`), CI + typed configs + real tests.
Superseded rather than ported: TorchRL loop → Mava; wfcrl env wrappers → the
engine itself; reward normalisation transform → engine-side reward, revisit
at M1 if training needs it. Still open from DiCoDe: real E(n)-equivariant
generator (T7), designer abstraction rebuilt JAX-side (T4).

---

## 3. Dependency plan

**Workspace (py3.13, jax 0.11 locked):** `windrl-engine` depends on
`jax>=0.5.3` (floor kept low for windrl-train interop), `jaxtyping`,
`pydantic>=2`; `matplotlib` behind its `viz` extra. `wind-rl` keeps `torch`
(cu130 index) for the generative stack only, plus `hydra-core`, `omegaconf`,
`pydantic-settings`, `wandb[media]`, `scipy` (SLSQP), `numpy`, `tqdm`.
`torchrl`/`tensordict`/`pettingzoo`/`wfcrl` are gone.

**windrl-train (workspace-excluded):** own py3.12 venv + lock; Mava pinned to
a git sha (jax==0.5.3 transitively). Revisit whenever Mava relaxes its pins —
the goal is to rejoin the workspace when jax constraints allow.

**CI** installs sequentially (CPU torch for wind-rl's generative closure, CPU
jax for windrl-engine) to avoid multi-GB CUDA wheels; the whole test suite
runs unfiltered (reference tests are golden-based, no wfcrl/MPI needed).
windrl-train is not exercised on CI yet (needs its own py3.12 job — open).

**Geometric-dependency rule (unchanged in spirit):** no `torch_scatter` /
`torch_cluster` ever; JAX-side the engine already uses dense `(N,N)`
interactions and `.at[].add()` scatters. Any future JAX GNN layers for
policies follow the same dense/topk pattern inside Mava network torsos.

---

## 4. Implementation task list (ordered, independently reviewable)

Each task: goal / files / interfaces / acceptance. Tasks 1–4 = walking skeleton.

**T1 — Package skeleton, config, settings, CI.** *(done)*
Goal: buildable package + tooling parity with author repos.
Files: `src/wind_rl/{config.py,scenario.py,experiment/settings.py,utils/*}`,
`docs/`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `tests/`.
Interfaces: `Config` base (pydantic, `extra="forbid"`, OmegaConf merge);
`WindRlSettings(BaseSettings)` with `WIND_RL_WDIR`.
Accept: `uv run mypy src` clean; `uv run pytest -q` green; CI job passes;
`Config.from_raw(OmegaConf.create({...}))` round-trips a scenario.

**T2 — Environment.** *(superseded — done differently)* Replaced by
`packages/windrl-engine`: JAX rewrite with golden-anchored FLORIS fidelity,
batched envs, layout as first-class input. Layout injection at reset (the
co-design seam) exists via `FarmLayout` but a designer-driven reset feed is
T4 work.

**T3 — MAPPO trainer smoke (walking skeleton).** *(superseded — done
differently)* Replaced by `packages/windrl-train` (Mava ff_mappo, continuous
actions). Smoke runs complete on turb3 and horns_rev2. Still open from the
original acceptance: an experiments-framework run with an asserted
power-increase threshold (M1 exit criterion).

**T4 — Designer abstraction + baseline designers (JAX-side rebuild).**
Goal: reintroduce the `Designer` interface over `windrl_engine`'s
`FarmLayout` (the torch implementation was deleted with the RL stack):
layout generation/update API, env reset feed, and the static/search
baselines (Random/Fixed/Manual/Sampling/Descent/Reinforce/Replay).
Accept: feasible `(B,N,2)` batches (min-distance + boundary); batched envs
consume designer layouts at reset; Manual matches a published HornsRev1
layout.

**T5 — Permutation-invariant policy/critic (Mava network torsos).**
Goal: DeepSets/attention torsos plugged into Mava's actor/critic network
config (JAX, dense/topk graphs — no compiled geometric extensions).
Accept: permutation-equivariance unit test; MAPPO on 8 turbines beats the
MLP torso at matched frames (thresholded).

**T6 — Diffusion reference designer + env critic + distillation.**
Goal: reproduce DiCoDe's guided-DDIM `DicodeDesigner` for comparison under our infra.
Files: `generative/{diffusion.py,guidance.py,constraints.py}`,
`design/value_learner.py`.
Interfaces: guided sampling with PUG projected guidance; `ValueLearner.update`.
Accept: on 8-turbine FLORIS, DicodeDesigner co-design matches published DiCoDe
power within tolerance; feasibility maintained; NFE/iteration logged.

**T7 — E(n)-equivariant generator + policy (C2).**
Goal: real EGNN replacing the DiCoDe stub; equivariant layout prior (torch,
generative stack) + equivariant policy torso (JAX, Mava).
Interfaces: E(n)-equivariant layers (dense scatters, both frameworks).
Accept: rotation/translation-equivariance unit tests pass; equivariant layout
prior + equivariant policy meets/beats T5 at matched compute; run is stable
**without guidance-weight annealing** (ablate annealing on/off).

**T8 — Scale to 32/64/92 + real farms (C1).**
Goal: co-design on large real layouts; wall-clock/NFE benchmarks.
Files: `experiments/0002_scale/{run.py,conf/}`, report + journal.
Interfaces: scenarios `wfcrl_{32,64,92}` + HornsRev1/2/Ormonde/WMR.
Accept: the equivariant co-designer trains stably at 64 and 91 turbines where
REINFORCE collapses; power > all non-generative baselines (thresholded).
(windrl-engine makes this cheap: 91T is ~1.9 ms/env-step batched on CPU.)

**T9 — Domain-specific fine-tuning (C3).**
Goal: fine-tune pretrained layout prior per site (boundary + wind rose).
Files: `experiments/0003_finetune/{run.py,conf/}`; `generative/diffusion.py` hooks.
Interfaces: fine-tune API on a pretrained prior with site constraint/wind-rose
conditioning; (stretch) FastFarm policy transfer — see T11 caveat.
Accept: fine-tuned prior yields higher-power feasible layouts than generic prior
on >=2 real sites (thresholded); (stretch) FLORIS->FastFarm policy fine-tune
beats from-scratch FastFarm at matched wall-clock.

**T10 — Paper experiment sweep + figures (M4).**
Goal: designers x scenarios x seeds sweep, ablations, evidence-gated reports.
Files: `experiments/0004_paper/{run.py,conf/}`, `report.md`.
Accept: all headline claims C1–C3 backed by asserted thresholds in code; figures
regenerate from logged runs; `report.md` states hypothesis->setup->results->
decision per the experiments contract.

**T11 (optional) — FastFarm high-fidelity integration.**
Goal: MPI/FastFarm path if C3's transfer track is adopted. Caveat: the wfcrl
fork (which carried the FastFarm MPI interface) left the repo 2026-07-23; this
task now means a fresh integration against windrl-engine's env contract.
Accept: FastFarm 3-turbine env runs a short rollout in a CI-skippable slow test.

---

## 5. Owner decisions (2026-07-19)

1. **C3 scope: layout-prior per-site fine-tuning is the headline.**
   FLORIS->FastFarm policy transfer is a stretch experiment only; FastFarm/MPI is
   NOT on the critical path (T11 stays optional).
2. **Architectures: research first, simple first.** A dedicated research pass on
   geometric/equivariant architectures is commissioned separately. The FIRST
   implemented version uses something simple (e.g. a plain GCN). There must be an
   **independent experiment suite just for quickly benchmarking architectures**
   (its own numbered experiments framework, cheap proxy tasks, fast turnaround)
   before any architecture is promoted into the main training pipeline. The
   long-term target is the leading spherical-harmonics-with-attention family,
   specialised to 2D (circular harmonics, cheap via FFT) — T5/T7 are subordinate
   to what the research pass + benchmark suite conclude.
3. **Baselines: deprioritised for now.** T6 (DiCoDe DDIM reference designer)
   drops out of the critical path; comparisons under our own infra can come later.
   Build the method first.

4. **Monorepo layout.** The main package lives at `packages/wind-rl/src/wind_rl/`
   (catan-engine layout): the root pyproject is a virtual workspace coordinator
   (tooling + dependency groups only); each package under `packages/` owns its
   dependencies and tests.
5. **Experiment 0001 is the fixed-layout MARL benchmark**: training various MARL
   agents (architectures/algorithm variants) on fixed wind-farm layouts.
   Co-design experiments start at 0002+.
6. **mypy at maximum feasible strictness** (disallow_any_generics on; extra
   error codes; explicit-Any minimized at third-party boundaries).

Remaining open (non-blocking, revisit at M3/M4): compute/fidelity budget for the
92-turbine sweep (FLORIS-only assumed sufficient until shown otherwise).
