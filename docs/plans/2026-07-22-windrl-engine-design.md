# windrl-engine — JAX wind-farm simulator design

Date: 2026-07-22 · Status: DRAFT (owner review required)

A from-scratch JAX reimplementation of the WFCRL/FLORIS wind-farm environment
with first-class batching (parallel envs × wind conditions) and layout as a
first-class input. Physics ground truth:
`docs/plans/2026-07-22-jax-windfarm-step-spec.md` (the "step spec") — full GCH
model, numerically validated against the WFCRL env. Structural template:
`catan-engine/packages/settlrl-engine` (design study in session scratchpad).

## Decisions

1. **Package**: `packages/windrl-engine/src/windrl_engine/`, own
   `pyproject.toml`, hatchling, `py.typed`. Runtime deps: `jax>=0.7`,
   `jaxtyping>=0.3.10`, `pydantic>=2` (config surface only). Extras:
   `viz = [matplotlib]`. No gymnasium / pettingzoo / torch anywhere in the
   package.
2. **PyTrees are plain `NamedTuple`s.** Static geometry (`FarmLayout`) separate
   from dynamic state (`FarmState`). `._replace` is the functional update. No
   flax/equinox.
3. **Single-farm pure cores; batch axes added at the edge** via
   `jax.jit(jax.vmap(...))`. Core shapes never carry a batch axis:
   `(turbines,)`, `(turbines, grid, grid)`. Batched surfaces add leading
   `(envs,)` and, for rose evaluation, `(conditions,)`.
4. **jaxtyping annotations are the shape documentation**, aliases pinned to
   domain constants beside them (`grid=3`). Runtime-enforced in tests only via
   beartype `install_import_hook` in `tests/conftest.py`. `mypy strict = true`
   (relax `disallow_any_generics`/`explicit` for JAX noise only if needed).
5. **float64.** The solve must run under `jax.config.update("jax_enable_x64",
   True)` for reference agreement (step spec §9). The package never mutates
   global JAX config; tests and consumers enable x64. Code is written
   dtype-following (no hardcoded `float32`).
6. **Layered imports, pointing down only**: `env → physics → farm`;
   `analysis → physics → farm`. No cycles.
7. **Pydantic v2 `Config`** (repo standard, `extra="forbid"`) only at the
   user-facing construction surface (`env/config.py`), converted once into
   static jit args + PyTrees. The functional core takes arrays and Python
   scalars, never pydantic objects.
8. **Multi-agent = the turbine axis.** WFCRL's AEC env advances state only
   after all N agents act, so a jointly-stepped parallel API is semantically
   equivalent: actions `(envs, turbines)`, per-turbine observations
   `(envs, turbines, ...)`, shared scalar reward broadcast per turbine.
9. **Branchless numerics**: compute near/far wake (and every masked branch)
   unconditionally and combine with `jnp.where`/mask multiplies — identical
   results to FLORIS's `if mask.sum()` guards (step spec §9), jit-safe.
10. **Visualization/analysis is a real subpackage** (`analysis/`), not
    tests-only: flow-field slices, wind-rose/AEP metrics, matplotlib plots
    behind the `viz` extra.

## Package tree

```
packages/windrl-engine/
  pyproject.toml
  src/windrl_engine/
    py.typed
    farm/
      turbine.py       # NREL-5MW constants + the three 50-pt tables (verbatim
                       #   from nrel_5MW.yaml) + Ct/Cp/inner-power interpolants
                       #   with exact fill/clip semantics (spec §5.5)
      layout.py        # FarmLayout NamedTuple (x, y); builders: procedural
                       #   row/grid layouts + reference layouts (Turb3 row,
                       #   Ablaincourt 7T, HornsRev2 91T — values from wfcrl
                       #   data_cases, copied as literals)
      wind.py          # WindCondition NamedTuple; reset sampling
                       #   (8·Weibull(8) clip [0,28]; Normal(270,20)%360,
                       #   spec §3); WindRose container + rose grids
      state.py         # FarmState NamedTuple (yaw, yaw_accumulator,
                       #   step_count, wind, key); make_state
    physics/
      frame.py         # rotation to wind-aligned frame (270°=west), rotor
                       #   grid construction (3×3, z=90+[-31.5,0,31.5]),
                       #   upstream argsort permutation (spec §5.1)
      flow.py          # initial flow field: shear profile u=ws·(z/90)^0.12,
                       #   dudz (spec §5.2)
      thrust.py        # cubic-mean rotor velocity, Ct lookup, axial
                       #   induction (spec §5.3 head)
      deflection.py    # Gauss deflection field + secondary steering
                       #   wake_added_yaw (spec §5.3)
      transverse.py    # calculate_transverse_velocity: 6 vortices,
                       #   mixing-length decay, W≥0 clamp (spec §5.3)
      deficit.py       # Gauss velocity deficit, near/far branchless
                       #   (spec §5.3)
      turbulence.py    # crespo_hernandez added TI, yaw-added recovery
                       #   mixing, WAT area-overlap integration (spec §5.3)
      solver.py        # sequential per-turbine lax.fori_loop in sorted
                       #   order, SOSFS hypot accumulation, unsort/finalize;
                       #   FlowSolution NamedTuple (spec §5.3–5.4)
      power.py         # turbine powers (W), load proxies (N,4), local wind
                       #   measurements (spec §6a–6c)
    env/
      config.py        # WindFarmEnvConfig (pydantic): layout or case name,
                       #   yaw control spec, horizon, load_coef, control mode
      actions.py       # duty-cycle limiter + delta→absolute pipeline
                       #   (spec §4), continuous and discrete variants
      env.py           # functional reset/step cores (single farm) +
                       #   BatchedWindFarmEnv: jit(vmap) surface, observations
                       #   (spec §6d), reward (spec §7), truncation (spec §8),
                       #   device-side auto-reset, lax.scan rollout
      spaces.py        # tiny local frozen-dataclass Box/MultiDiscrete
                       #   descriptors (no gym dep)
    analysis/
      flow_viz.py      # horizontal/vertical flow-slice evaluation at query
                       #   points (re-runs wake models with per-turbine
                       #   quantities from the rotor solve)
      metrics.py       # rose-batched power surface, AEP, wake-loss fraction
      plots.py         # matplotlib: layout map, flow-field heatmap, wind
                       #   rose, power-vs-direction (viz extra)
  tests/               # written by the independent test team
    conftest.py        # x64 enable + beartype import hook (physics/farm
                       #   modules only — batched modules excluded)
```

## Boundary signatures (fixed — teams code against these)

```python
# farm/layout.py
class FarmLayout(NamedTuple):
    x: Float[Array, "turbines"]   # meters, world frame
    y: Float[Array, "turbines"]

# farm/wind.py
class WindCondition(NamedTuple):
    speed: Float[Array, ""]       # m/s freestream
    direction: Float[Array, ""]   # deg, 270 = wind from west

def sample_wind(key: Key[Array, ""]) -> WindCondition

# farm/state.py
class FarmState(NamedTuple):
    yaw: Float[Array, "turbines"]              # absolute deg, [-40, 40]
    yaw_accumulator: Float[Array, "turbines"]  # Σ|applied Δyaw| deg
    step_count: Int[Array, ""]
    wind: WindCondition
    key: Key[Array, ""]

# physics/solver.py
class FlowSolution(NamedTuple):
    u: Float[Array, "turbines grid grid"]      # m/s, original turbine order
    v: Float[Array, "turbines grid grid"]
    w: Float[Array, "turbines grid grid"]
    turbulence_intensity: Float[Array, "turbines"]

# farm/turbine.py — turbine library as a PyTree (single shared spec)
class TurbineSpec(NamedTuple):
    rotor_diameter: float; hub_height: float; pP: float; tsr: float
    generator_efficiency: float; ref_density: float
    wind_speed_table: TurbineTable; thrust_table: TurbineTable
    power_table: TurbineTable; power_scale: float          # power(u) = interp(u)·power_scale
    ct_fill_low: float; ct_fill_high: float
def nrel5mw_v3() -> TurbineSpec                             # FLORIS 3.5 (default)
def nrel5mw_v4() -> TurbineSpec                             # FLORIS 4.6.6 (cosine-loss, abs-kW)

def solve_farm(
    layout: FarmLayout, wind: WindCondition, yaw: Float[Array, "turbines"],
    *, fidelity: str = "floris", turbine: TurbineSpec = nrel5mw_v3(),
) -> FlowSolution
    # fidelity is a STATIC argname (two jit specializations): "corrected" drops the
    # rotor-plane ULP self-interaction and the stale-TI deflection ordering.

# physics/power.py
def turbine_powers(
    u: Float[Array, "turbines grid grid"], yaw: Float[Array, "turbines"],
    *, turbine: TurbineSpec = nrel5mw_v3(),
) -> Float[Array, "turbines"]                  # Watts
def load_proxies(solution: FlowSolution) -> Float[Array, "turbines 4"]
def local_wind(
    solution: FlowSolution, wind: WindCondition
) -> tuple[Float[Array, "turbines"], Float[Array, "turbines"]]  # speed, dir

# env/env.py — single-farm functional core
class Observation(NamedTuple):
    yaw: Float[Array, "turbines"]
    freewind: Float[Array, "2"]                # [speed, direction]
    wind_speed: Float[Array, "turbines"]       # local ∛(mean u³)
    wind_direction: Float[Array, "turbines"]

def reset(
    layout: FarmLayout, key: Key[Array, ""],
    wind: WindCondition | None = None,
) -> tuple[FarmState, Observation]
def reset(
    layout: FarmLayout, key: Key[Array, ""], wind: WindCondition | None = None,
    *, fidelity: str = "floris", turbine: TurbineSpec = nrel5mw_v3(),
) -> tuple[FarmState, Observation]
def step(
    layout: FarmLayout, state: FarmState, action: Float[Array, "turbines"],
    *, yaw_step: float, load_coef: float, horizon: int,
    fidelity: str = "floris", turbine: TurbineSpec = nrel5mw_v3(),
) -> tuple[FarmState, Observation, Float[Array, ""], Bool[Array, ""]]
    # (state', obs, reward, truncated). fidelity static; turbine a runtime PyTree arg.
    # WindFarmEnvConfig exposes both as fidelity: {"floris","corrected"} and
    # turbine: {"nrel5mw_v3","nrel5mw_v4"}.
```

`BatchedWindFarmEnv` wraps these with `jit(vmap(...))` over a leading `envs`
axis (shared layout; per-env state), constructed from `WindFarmEnvConfig`.
`analysis.metrics` vmaps `solve_farm` over a `conditions` axis for rose
evaluation.

## Testing strategy (independent team)

- **Differential oracle** (the backbone, marked `sim`, excluded from CI):
  drive `wfcrl` env and `windrl_engine` with identical layout, wind overrides
  (`options={"wind_speed": …, "wind_direction": …}`), and action streams;
  assert per-step agreement on powers, observations, and reward — `<1e-8`
  relative under x64 (step spec §9). Component-level: compare `FlowSolution.u`
  directly against `fi.floris.flow_field.u` on fixed cases.
- **Invariant tests** (CI-safe, no wfcrl): turbine-permutation equivariance,
  rotation invariance (rotate layout + wind direction together), batched ==
  stacked-singles under vmap, deficit ≥ 0, power ≤ rated, duty-cycle limiter
  algebra, deterministic replay from (seed, actions).
- **Shape/dtype enforcement**: beartype import hook over single-farm modules.
- Test names state behavior; assert real values (coding guidelines).

## Build order

1. **I0 — scaffolding + `farm/`** (opus): package skeleton, pyproject,
   workspace integration, full `farm/` implementation (tables verbatim),
   `physics/`+`env/`+`analysis/` stubs carrying the exact signatures above.
2. **Parallel**: I1 `physics/` (opus, the heart), I2 `env/` (opus),
   I3 `analysis/` (sonnet). Disjoint paths; stubs make cross-imports
   typecheck before integration.
3. **Independent team**: T1 differential tests (opus), T2 invariant tests
   (sonnet), R1 spec-conformance + guidelines review (opus). Tests are written
   from the step spec and the WFCRL reference, not from reading the
   implementation.
4. Orchestrator integrates, runs the four repo checks, commits. Implementation
   agents do not commit (avoids parallel index races).
