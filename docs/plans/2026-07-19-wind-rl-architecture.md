# wind-rl — Architecture & Research Plan

Date: 2026-07-19 · Author: planning agent · Status: DRAFT (owner review required)

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
- **M1 — Reproduce baseline.** MAPPO-on-WFCRL at small scale (2–3 then 8–16
  turbines, FLORIS, yaw). Match DiCoDe's power-capture numbers. Walking
  skeleton + infra (CI, tests, config, logging).
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

Design within the **uv workspace**. The root pyproject is a virtual workspace
coordinator; the main package is `packages/wind-rl/` (src layout).
`packages/wfcrl-env` is the author's fork (submodule) consumed as a library.
`experiments/` holds numbered frameworks following the
physics-informed-flow-map convention: `NNNN_slug/run.py` + `conf/`,
`report.md`, append-only `JOURNAL.md`.

### `wind_rl` package submodule tree (one-line responsibilities)

```
packages/wind-rl/src/wind_rl/
  config.py          # pydantic v2 Config base (extra="forbid") + OmegaConf/Hydra
                     #   override merge — the "pydra" pattern. Discriminated-union
                     #   configs for designer/model/scenario.
  scenario.py        # ScenarioConfig (n_turbines, max_steps, map_x/y_length,
                     #   min_distance) + real-farm registry mapping name->wfcrl case.
  env/
    windfarm.py      # DesignableWindFarmEnv(MAWindFarmEnv): adds `layout` to
                     #   state/obs; reset(options={xcoords,ycoords}) rebuilds MDP.
    wrapper.py       # WfcrlCoDesignWrapper(PettingZooWrapper): pulls layout from
                     #   designer at _reset; injects state() into td.
    factory.py       # make_env(mode, scenario, designer, simulator, device) ->
                     #   TransformedEnv; the single env-creation pipeline.
    transforms.py    # RewardNormalisation TorchRL Transform (fixes DiCoDe TODO:
                     #   normalisation was a script, now an env transform), RewardSum.
    render.py        # matplotlib rgb_array layout render for wandb video.
  models/
    base.py          # Policy/Critic/LayoutGenerator/EnvCritic protocols.
    gnn.py           # Permutation-invariant GNN policy/critic (DeepSets/PNA).
    equivariant.py   # E(n)-equivariant layout generator + policy (EGNN); the real
                     #   architecture replacing DiCoDe's E3GNN stub.
    mlp.py           # MLP baselines (parity with DiCoDe model_type="mlp").
    heads.py         # MultiAgentMLP + NormalParamExtractor -> TanhNormal actor;
                     #   mean-pooled centralized critic head.
  generative/
    diffusion.py     # DDIM layout generator (repro of DiCoDe designer).
    guidance.py      # Guided sampling: PUG-style projected guidance + constraint
                     #   projection (min-distance, boundary polygon).
    constraints.py   # Feasibility: min-distance + site boundary; soft penalty and
                     #   hard (SLSQP) projection.
  design/
    base.py          # Designer[SC] ABC: generate_layout_batch(n), update(td),
                     #   get_state/get_logs; DesignProducer/DesignConsumer buffer.
    buffer.py        # File-backed lock-protected layout buffer (pop at env reset).
    value_learner.py # ValueLearner: critic distillation for env-critic training.
    designers.py     # Random/Fixed/Manual/Sampling/Descent/Reinforce/Replay +
                     #   DicodeDesigner; create_designer() factory
                     #   over discriminated-union DesignerConfig.
  rl/
    mappo.py         # MAPPO trainer (TorchRL ClipPPOLoss/GAE/SyncDataCollector).
    trainer.py       # Trainer class: collect->GAE->PPO epochs->designer.update->
                     #   eval->checkpoint loop. De-generic'd from DiCoDe's ABC.
  experiment/
    harness.py       # Run lifecycle: start_run/log/finish (wandb), checkpointing,
                     #   workdir resolution via pydantic-settings.
    settings.py      # WindRlSettings(BaseSettings): WIND_RL_WDIR etc. (fixes
                     #   DiCoDe's hardcoded ~/.diffusion_co_design + hardcoded paths).
    normalisation.py # Compute NormalisationStatistics from random-policy rollouts.
  utils/             # seeding, device management, logging helpers.
```

### Key interfaces / signatures

**Designer** (fix DiCoDe: no dummy ScenarioConfig, buffer stays):
```python
class Designer(Protocol[SC]):
    def generate_layout_batch(self, batch_size: int) -> np.ndarray:  # (B, N, 2)
    def update(self, sampling_td: TensorDict) -> None: ...            # no-op for static
    def to_td_module(self) -> TensorDictModule: ...   # reset_policy for env wrapper
    def get_logs(self) -> dict: ...

def create_designer(cfg: DesignerConfig, scenario: SC, artifact_dir: Path,
                    device: str) -> tuple[Designer[SC], Callable[[], DesignConsumer]]:
```

**Env pipeline** (single entry point; simulator selectable — fixes FLORIS-hardcode):
```python
def make_env(mode: Literal["train","eval","reference"], scenario: ScenarioConfig,
             designer: DesignConsumer, simulator: Literal["floris","fastfarm"]="floris",
             device: str | None = None, render: bool = False) -> TransformedEnv:
```
Internally: build `FlorisCase`/`FastFarmCase` from scenario coords ->
`DesignableWindFarmEnv(interface=FlorisInterface|FastFarmInterface,
controls=get_default_control(["yaw"]), ...)` -> `aec_to_parallel` ->
`WfcrlCoDesignWrapper(reset_policy=designer.to_td_module())` ->
`TransformedEnv(Compose(RewardNormalisation(...), RewardSum(...), RemoveEmptySpecs()))`.

**Trainer**:
```python
class MappoTrainer:
    def __init__(self, cfg: TrainingConfig, project_name: str): ...
    def run(self) -> None: ...   # collect frames_per_batch -> minibatch GAE ->
                                 # PPO n_epochs x n_mini_batches -> designer.update ->
                                 # periodic eval rollout + wandb video + checkpoint
```

### What to keep vs fix (from DiCoDe)
Keep: TorchRL MAPPO loop, `Designer` abstraction, file-backed layout buffer,
critic distillation `ValueLearner`, `create_designer`/`create_env` factories,
two-stage config with scenario registry, wandb video/artifact logging.
Fix (DiCoDe TODOs): (a) reward normalisation as a **TorchRL env Transform**, not
a standalone script; (b) **no dummy `ScenarioConfig`** in generation code; (c)
**no hardcoded paths** — workdir via `pydantic-settings` (`WIND_RL_WDIR`); (d)
simulator is a config field (wire FastFarm); (e) real E(n)-equivariant generator
replacing the empty `E3GNN` stub; (f) CI + typed configs + real tests.

### torchrl migration risk (FLAG)
DiCoDe pins `torchrl>0.9,<0.10`; wind-rl is `torchrl==0.11.1`. Expect breaking
changes across two minor versions in: `PettingZooWrapper`, `ClipPPOLoss`
(`deactivate_vmap`/`set_keys` args), `SyncDataCollector`, GAE `value_estimator`,
`ProbabilisticActor`/`TanhNormal` key wiring, `TransformedEnv` spec transforms.
**M1 task 2 must include a torchrl-0.11 smoke port before any algorithm work**;
budget time for API churn. `support_vmap=False` path (PNA GAE) especially at risk.

---

## 3. Dependency plan

Add to root `pyproject` `[project.dependencies]` (another agent edits — this plan
only specifies): `hydra-core`, `omegaconf`, `pydantic>=2`, `pydantic-settings`,
`tensordict` (matched to torchrl 0.11.1), `wandb[media]`, `matplotlib`, `numpy`,
`scipy` (SLSQP projection), `tqdm`. `floris` and `fastfarm`/MPI deps come via
`wfcrl-env`'s own extras — do not duplicate.

Dev `[dependency-groups]` (match author convention): `ruff>=0.15`, `mypy>=1.11`,
`pytest>=8`, `pytest-xdist`, `pre-commit>=4`, `ipykernel`. Local pre-commit hooks
(ruff-check --fix, ruff-format, mypy on `src`+`tests`+`experiments`, fast pytest)
and a `.github/workflows/ci.yml` mirroring catan-engine (`uv sync --locked`, ruff
check/format --check, mypy, pytest, experiments smoke with `WANDB_MODE=disabled`).

**Avoid DiCoDe's torch_scatter/torch_cluster manual `--no-build-isolation` pain.**
Two options, prefer (A):
- **(A) Architectures that need neither.** Implement DeepSets/attention-based
  permutation-invariant layers and a hand-rolled EGNN using dense/`torch`-native
  `scatter_add`/`index_add_` and full or KNN graphs via `torch.cdist` +
  `topk`. At N<=92 dense O(N^2) message passing is cheap; no compiled geometric
  extensions required. **This is the recommended default.**
- **(B) If PyG is desired**, add `torch-geometric` **only** (pure-Python core;
  needs no scatter/cluster wheels for the layers we use) and forbid any op
  requiring `torch-scatter`/`torch-cluster`. Document the constraint in the
  package README so nobody reintroduces the wheel-build pain.

torch/torchvision already routed to the cu130 index in `pyproject`; keep that.

---

## 4. Implementation task list (ordered, independently reviewable)

Each task: goal / files / interfaces / acceptance. Tasks 1–4 = walking skeleton.

**T1 — Package skeleton, config, settings, CI.**
Goal: buildable package + tooling parity with author repos.
Files: `src/wind_rl/{config.py,scenario.py,experiment/settings.py,utils/*}`,
`docs/`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `tests/`.
Interfaces: `Config` base (pydantic, `extra="forbid"`, OmegaConf merge);
`WindRlSettings(BaseSettings)` with `WIND_RL_WDIR`.
Accept: `uv run mypy src` clean; `uv run pytest -q` green; CI job passes;
`Config.from_raw(OmegaConf.create({...}))` round-trips a scenario.

**T2 — Env wrapper + torchrl-0.11 port.**
Goal: `DesignableWindFarmEnv` + `WfcrlCoDesignWrapper` + `make_env` on FLORIS,
ported to torchrl 0.11.1.
Files: `env/{windfarm.py,wrapper.py,factory.py,transforms.py,render.py}`.
Interfaces: `make_env(...)` (§2); `RewardNormalisation` Transform.
Accept: test builds a 3-turbine FLORIS env, `reset(options={xcoords,ycoords})`
rebuilds the MDP and `state()["layout"]` reflects new coords; a random rollout of
`scenario.max_steps` runs without spec/vmap errors; obs/action specs match
`num_turbines`.

**T3 — MLP MAPPO trainer smoke (walking skeleton).**
Goal: end-to-end MAPPO training on 2–3 turbine FLORIS with `FixedDesigner`.
Files: `models/{mlp.py,heads.py}`, `rl/{mappo.py,trainer.py}`,
`experiments/0001_mappo_smoke/{run.py,conf/}`, `report.md`, `JOURNAL.md`.
Interfaces: `MappoTrainer.run()`; TorchRL `ClipPPOLoss`+GAE+`SyncDataCollector`.
Accept: `WANDB_MODE=disabled` 5-iter run completes; asserted mean episode power
strictly increases over a short run (thresholded, in code); checkpoint written +
reloadable.

**T4 — Designer abstraction + buffer + baseline designers.**
Goal: `Designer` interface, layout buffer, and the static/search baselines.
Files: `design/{base.py,buffer.py,designers.py}`; `scenario.py` real-farm registry.
Interfaces: `Designer` protocol (§2); `create_designer(cfg, ...)`.
Accept: Random/Fixed/Manual/Sampling/Descent/Reinforce/Replay each produce a
feasible `(B,N,2)` batch (min-distance + boundary satisfied); env pops layouts at
reset; `create_designer` dispatches over the discriminated union; Manual matches a
published HornsRev1 layout.

**T5 — Permutation-invariant GNN policy/critic.**
Goal: DeepSets/PNA policy+critic (no torch-scatter), parity with DiCoDe GNN.
Files: `models/{gnn.py,base.py}`.
Interfaces: `Policy`/`Critic` protocols; KNN/full graph via `torch.cdist`.
Accept: permutation-equivariance unit test (permuting turbines permutes actions
identically); MAPPO run on 8-turbine FLORIS beats MLP at matched frames
(thresholded); no compiled geometric extension imported.

**T6 — Diffusion reference designer + env critic + distillation.**
Goal: reproduce DiCoDe's guided-DDIM `DicodeDesigner` for comparison under our infra.
Files: `generative/{diffusion.py,guidance.py,constraints.py}`,
`design/value_learner.py`.
Interfaces: guided sampling with PUG projected guidance; `ValueLearner.update`.
Accept: on 8-turbine FLORIS, DicodeDesigner co-design matches published DiCoDe
power within tolerance; feasibility maintained; NFE/iteration logged.

**T7 — E(n)-equivariant generator + policy (C2).**
Goal: real EGNN replacing the DiCoDe stub; equivariant layout prior + policy.
Files: `models/equivariant.py`.
Interfaces: E(n)-equivariant layers (torch-native scatter).
Accept: rotation/translation-equivariance unit tests pass; equivariant layout
prior + equivariant policy meets/beats T5 at matched compute; run is stable
**without guidance-weight annealing** (ablate annealing on/off).

**T8 — Scale to 32/64/92 + real farms (C1).**
Goal: co-design on large real layouts; wall-clock/NFE benchmarks.
Files: `experiments/0002_scale/{run.py,conf/}`, report + journal.
Interfaces: scenarios `wfcrl_{32,64,92}` + HornsRev1/2/Ormonde/WMR.
Accept: the equivariant co-designer trains stably at 64 and 92 turbines where
REINFORCE collapses; power > all non-generative baselines (thresholded); FLORIS
case-file housekeeping bounded (no unbounded `__simul__` accumulation).

**T9 — Domain-specific fine-tuning (C3).**
Goal: fine-tune pretrained layout prior per site (boundary + wind rose).
Files: `experiments/0003_finetune/{run.py,conf/}`; `generative/diffusion.py` hooks.
Interfaces: fine-tune API on a pretrained prior with site constraint/wind-rose
conditioning; optional FastFarm policy-transfer path in `make_env(simulator=...)`.
Accept: fine-tuned prior yields higher-power feasible layouts than generic prior
on >=2 real sites (thresholded); (stretch) FLORIS->FastFarm policy fine-tune
beats from-scratch FastFarm at matched wall-clock.

**T10 — Paper experiment sweep + figures (M4).**
Goal: designers x scenarios x seeds sweep, ablations, evidence-gated reports.
Files: `experiments/0004_paper/{run.py,conf/}`, `report.md`.
Accept: all headline claims C1–C3 backed by asserted thresholds in code; figures
regenerate from logged runs; `report.md` states hypothesis->setup->results->
decision per the experiments contract.

**T11 (optional) — FastFarm high-fidelity integration hardening.**
Goal: robust MPI/FastFarm path if C3's transfer track is adopted.
Accept: FastFarm 3-turbine env runs a short rollout in CI-skippable slow test.

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
