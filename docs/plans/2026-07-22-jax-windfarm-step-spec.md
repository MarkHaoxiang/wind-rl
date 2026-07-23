# JAX WFCRL/FLORIS wind-farm step — ground-truth spec

Reference implementation: WFCRL fork
(`packages/wfcrl-env/wfcrl/{simple_env,multiagent_env,mdp,interface}.py`) driving
FLORIS **v3.5** (`.venv/.../floris/simulation/`). This document specifies one
environment `step` and `reset` exactly enough to reproduce them numerically in
JAX. Every constant is copied from the shipped configs; every equation is
transcribed from source. Line references are to the FLORIS 3.5 install and the
WFCRL fork as of this commit.

The physics core is `sequential_solver` (velocity model `gauss` dispatches
there, *not* `cc_solver`) with the full GCH extras enabled: secondary steering,
yaw-added recovery, transverse velocities.

---

## 1. Scope

One `WindFarmMDP.take_action` = one `FlorisInterface.update_command`:

1. advance wind time-series cursor (constant wind for HornsRev2 → no-op),
2. `fi.reinitialize` **iff** wind speed or direction changed,
3. `fi.calculate_wake(yaw_angles)` with `yaw` shape `(1,1,N)`,
4. read powers, loads, local wind.

FLORIS canonical array shape is `(n_wd, n_ws, n_turbines, n_grid, n_grid)` with
`n_grid = 3` (3×3 rotor grid). WFCRL always runs **`n_wd = n_ws = 1`**. The JAX
port should keep these two leading axes as the natural batch axes:

- `n_ws`/`n_wd` axis → batch over wind conditions (a farm evaluated at many
  `(speed, dir)` simultaneously). Everything downstream of grid construction is
  elementwise in these axes **except** the per-wind-direction turbine sort
  order (§9), which is static per `wd`.
- An additional outer `n_env` axis → batch over parallel env instances (each
  with its own yaw/duty-cycle/PRNG state, §2). The wake solve is `vmap`-able
  over this axis provided the layout is shared; if layout differs per env, the
  argsort permutation also becomes per-env.

This spec describes the **HornsRev2-like case**: fixed layout, one wind
condition per env, full GCH. `n_turbines = 91` for HornsRev2 (layout in
`data_cases.py` `floris_hornsrev2`; the `HornsRev2_` registry entry uses that
list — 91 coordinates).

---

## 2. State carried between steps

Per env instance (numpy dtypes in the reference; JAX should use `float64` — §9):

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `yaw_abs` | `(N,)` | float64 | absolute yaw command (deg), clipped to `[-40,40]`. This is `state["yaw"]`. |
| `wind_speed_free` | scalar | float64 | freestream speed `v∞` (m/s), fixed per episode unless time-series. |
| `wind_dir_free` | scalar | float64 | freestream direction (deg), `270`=wind from west. |
| `actuation_accumulator["yaw"]` | `(N,)` | float32 | running sum of `|applied Δyaw|` (deg), reset each episode. |
| `num_moves` | scalar int | — | steps since reset (central env); MA uses per-agent `_num_steps`. |
| `ts_cursor` | int | — | index into wind time-series (only if `wind_time_series`); HornsRev2 has none. |
| PRNG | — | — | `np.random.default_rng(seed)` for wind sampling at reset; global `np.random` for time-series start offset. Port: split a JAX key per reset. |

`state` dict (the observation, §6) additionally holds the *measured*
per-turbine fields, but those are recomputed each step from the flow field and
need not persist. `yaw_abs` is the only control field that persists and is
**not** read back from FLORIS — it is the integrated action (mdp.py:276-277
updates only `measures`, never controls).

Reference dtype notes: `mdp._cast_dict_array` casts state to float32 before the
transition, then `clip_to_dict_space` runs against a float64 Box. The stored
`start_state`/observation is therefore effectively float32-rounded then held as
float64. Powers/loads flow through float32 (`avg_powers` → `float32`;
`local_load_proxies` from float64 flow field). The JAX port may keep everything
float64 (deviations §10) — the float32 hops are lossy but their effect is
< 1e-6 relative and not part of the physics.

---

## 3. Reset semantics

`WindFarmMDP.reset(seed, options)` (mdp.py:230-268):

```
rng = np.random.default_rng(seed)
# wind speed: sampled unless set_wind_speed or wind_time_series
wind_speed = 8 * rng.weibull(8)          # a=8 shape param
wind_speed = clip(wind_speed, 0, 28)     # DEFAULT_BOUNDS["wind_speed"]
# wind direction: sampled unless set_wind_direction or wind_time_series
wind_direction = rng.normal(270, 20) % 360
wind_direction = clip(wind_direction, 0, 360)   # DEFAULT_BOUNDS["wind_direction"]
```

`options["wind_speed"]` / `options["wind_direction"]` override the sample when
present. For `FlorisCase` (`data_cases.py:82-100`) `set_wind_speed=False`,
`set_wind_direction=False`, `wind_time_series=None` → **both are sampled**. The
Weibull draw order matters for seed-exact reproduction: speed is drawn **before**
direction from the same `rng`.

`np.random.default_rng(8).weibull(8)` semantics: `weibull(a)` returns
`(-ln U)^(1/a)`, `U~Uniform(0,1)`; multiply by 8. JAX port must match numpy's
Generator stream if seed-exact resets are required (otherwise sample equivalently
and accept statistical, not bitwise, agreement).

**Warm-up burn-in**: after `interface.init(ws,wd)`, run
`for _ in range(start_iter + 1): interface.update_command()` (mdp.py:258-259).
`start_iter` defaults to `0` → **1 burn-in step** with `yaw = 0`
(`_current_yaw_command` initialised to zeros in `FlorisInterface.init`,
interface.py:609; `update_command()` with no arg leaves it zero). The burn-up
step's only effect for FLORIS is that the flow field / measurements are populated
before the first agent action; the steady-state solve is stateless so a single
zero-yaw solve suffices.

**Random time-series offset**: only when `wind_time_series` is a path/array
(interface.py:514-516): `start = np.random.randint(0, T)`; series is rolled by
`start`. HornsRev2 default has none.

**Initial yaw**: `yaw_abs = 0` for all turbines (burn-in ran zero yaw; the reset
start_state reads `get_measure("yaw")` = `fi.floris.farm.yaw_angles` which is the
last commanded zero vector).

**FLORIS `dt` / `t_init`**: `FlorisInterface.dt = 60` hardcoded
(interface.py:486). `FlorisCase.t_init = 0` (`data_cases.py`). ⚠ The prompt's
"t_init=300s" is a FastFarm value; for FLORIS `t_init` is unused and `dt=60`
only matters for the duty-cycle limiter denominator (§4). Flag for owner.

---

## 4. Action pipeline

Per step, the env receives `actions = {"yaw": Δ (N,)}` (delta actions).

### 4a. Duty-cycle limiter (applied *before* the transition)

Central env (simple_env.py:64-72), per control in `ACTUATORS_RATE`
(`{"yaw":0.3, "pitch":8}` deg/s):

```
num_moves += 1
actuating_time  = accumulated_actions["yaw"] / 0.3     # (N,) seconds of actuation
actuating_frac  = actuating_time / num_moves / dt      # dt = 60
actions["yaw"][actuating_frac >= 0.1] = 0.0            # zero the over-active turbines
```

`accumulated_actions` is the running `Σ|applied Δyaw|` **from previous steps**
(state var, §2). A turbine that has been commanded to move more than 10 % of the
elapsed wall-clock time (`num_moves * dt` seconds) gets its current action
zeroed. Note `0.3` is deg/s so `accumulated/0.3` converts summed degrees to
seconds of slewing.

MA env (multiagent_env.py:196-207) is identical but per-agent with the agent's
own `_num_steps[agent]` and accumulator scalar.

### 4b. Delta → absolute (`get_controlled_state_transition`, mdp.py:288-316)

```
state = clip(cast_float32(state), state_space)          # yaw box [-40,40]
if continuous_control:                                   # Box variant
    cmd = clip(Δ, action_space.low, action_space.high)   # low/high = ∓ controls["yaw"][2]
else:                                                    # MultiDiscrete variant
    cmd = (Δ - 1) * controls["yaw"][-1]                  # Δ∈{0,1,2}→{-step,0,+step}
yaw_abs = clip(state["yaw"] + cmd, -40, 40)              # state_space["yaw"] bounds
accumulator["yaw"] += |cmd|                              # for next step's limiter
```

Control spec `controls = {"yaw": (low, high, step)}`. For continuous control the
**action bound** is `controls["yaw"][2]` (the "step" slot is reused as the
per-step Box half-range); e.g. `("yaw", (-40,40,5))` → Box actions in `[-5,5]`.
For MultiDiscrete, `step = controls["yaw"][2]` and `Δ∈{0,1,2}` maps to
`{-step, 0, +step}`.

Absolute yaw clip bounds are always `DEFAULT_BOUNDS["yaw"] = [-40, 40]`
(mdp.py:52), independent of `controls`.

Sign/units: `yaw_abs` is degrees, passed as-is to FLORIS
`fi.calculate_wake(yaw_angles=yaw_abs[None,None,:])` (interface.py:561-563). No
radians conversion for FLORIS (unlike the MPI interface).

---

## 5. The wake solve, equation by equation

All of §5 is `sequential_solver` (solver.py:63-261) + the models it calls.
Arrays are `(1,1,N,3,3)`; the turbine axis (2) is **sorted upstream→downstream**.

### 5.0 Constants (copied verbatim; do NOT re-derive)

From WFCRL `simulators/floris/inputs/template/case.yaml`:

| Group | Param | Value |
|---|---|---|
| flow_field | `air_density` | 1.225 |
| | `reference_wind_height` | −1 → **replaced by hub height 90.0** (§5.2) |
| | `turbulence_intensity` (ambient TI) | 0.06 |
| | `wind_shear` | 0.12 |
| | `wind_veer` | 0.0 |
| wake flags | secondary_steering / yaw_added_recovery / transverse_velocities | all **true** |
| deflection `gauss` | `ad` | 0.0 |
| | `bd` | 0.0 |
| | `alpha` | 0.58 |
| | `beta` | 0.077 |
| | `dm` | 1.0 |
| | `ka` | 0.38 |
| | `kb` | 0.004 |
| velocity `gauss` | `alpha,beta,ka,kb` | 0.58, 0.077, 0.38, 0.004 |
| turbulence `crespo_hernandez` | `initial` | 0.1 |
| | `constant` | **0.5** (overrides FLORIS class default 0.9) |
| | `ai` | 0.8 |
| | `downstream` | −0.32 |
| combination | `sosfs` | (no params) |

Hardcoded in the GCH source (not config-exposed): `eps_gain = 0.2`
(deflection/gauss.py:285,381), `gch_gain = 2` for yaw-added recovery
(solver.py:199), mixing-length `lmda = D/8`, `kappa = 0.41`
(deflection/gauss.py:407-408), `BaseModel.NUM_EPS = 0.001` (base.py:77),
wake-added-turbulence `downstream_influence_length = 15·D`, lateral gate
`|Δy| < 2·D`, area-overlap / WAT thresholds `> 0.05` (solver.py:231-251).

From `turbine_library/nrel_5MW.yaml` (all turbines identical):

| Param | Value |
|---|---|
| `rotor_diameter` D | 126.0 |
| `hub_height` HH | 90.0 |
| `pP` | 1.88 |
| `pT` | 1.88 |
| `TSR` | 8.0 |
| `generator_efficiency` | 1.0 |
| `ref_density_cp_ct` | 1.225 |
| `ref_tilt_cp_ct` | 5.0 |
| `power_thrust_table.wind_speed` | 50-pt table, copy verbatim (yaml lines 148-199) |
| `power_thrust_table.thrust` (Ct) | 50-pt table (yaml lines 96-147) |
| `power_thrust_table.power` (Cp) | 50-pt table (yaml lines 43-95) |
| `multi_dimensional_cp_ct` | False |

The three 50-element tables MUST be copied verbatim into the port; they define
`fCt_interp`, `fCp_interp`, `power_interp` (§5.5, §6).

Tilt is inert: `set_tilt_to_ref_tilt` sets `tilt_angle = ref_tilt_cp_ct = 5.0`
everywhere, `correct_cp_ct_for_tilt = False`, so every `cosd(tilt − ref_tilt) =
cosd(0) = 1` and tilt drops out. The port may omit tilt entirely.

### 5.1 Grid construction (grid.py `TurbineGrid.set_grid`, 164-291)

Wind-direction convention: `270°` = wind from the west (flow in +x). Rotation
into wind-aligned frame (`utilities.rotate_coordinates_rel_west`, 222-270):

```
wdev = (wind_direction - 270) % 360            # deviation from west, degrees
xc = (min(x)+max(x))/2 ;  yc = (min(y)+max(y))/2   # center of bounding box
x' = (x-xc)·cos(wdev) - (y-yc)·sin(wdev) + xc
y' = (x-xc)·sin(wdev) + (y-yc)·cos(wdev) + yc
z' = z            (= 0 at tower base; hub added via HH in flow, not here)
```
`cos`/`sin` take degrees. `x'` increases downstream.

Rotor grid (3×3), radius ratio `0.5`, `disc_area_radius = 0.5·D/2 = 31.5`:
```
disc_grid = linspace(-31.5, 31.5, 3) = [-31.5, 0, 31.5]     # per turbine
_x[...,a,b] = x'                                            # constant over rotor plane
_y[...,a,b] = y' + disc_grid[a]      (spanwise index a on axis -2)
_z[...,a,b] = z' + disc_grid[b]        (vertical index b on axis -1)
```
Precisely (grid.py:263-264): `_y = y'[...,None,None] + disc_grid[None,None,:,:,None]`
and `_z = z'[...,None,None] + disc_grid[:,None]·ones(N,3,3)`. i.e. the spanwise
offset varies along axis −2, the vertical offset along axis −1.

**z' is the turbine hub height 90.0**, NOT the tower base:
`Farm.construct_coordinates` (farm.py:278-281) builds each `Vec3([x, y,
hub_height])`, so the coordinate z-column passed to `rotate_coordinates_rel_west`
is `HH`, and rotation leaves z unchanged (`z' = z = 90`). Grid z therefore spans
`HH + disc_grid = [58.5, 90, 121.5]`. This is what makes the shear profile
`u_initial = ws·(z/HH)^0.12` (§5.2) well-defined (z>0) and centers the rotor on
the hub. Resolved — no z=0 tower-base ambiguity; use `z_grid = 90 + [-31.5,0,31.5]`.

Sorting (grid.py:271-280): `sorted_indices = _x.argsort(axis=2)` (stable
ascending; numpy default quicksort is **not stable** — ties broken by
implementation). `x_sorted = take_along_axis(_x, sorted_indices, 2)` (same for
y,z). `unsorted_indices = sorted_indices.argsort(axis=2)` restores original
order at finalize. For the JAX port: precompute one permutation per wind
direction (§9); ties are astronomically unlikely for real layouts but must use
the same argsort kind if bitwise agreement is required.

### 5.2 Initial flow field (flow_field.py `initialize_velocity_field`, 118-227)

```
wind_profile(z) = (z / reference_wind_height) ** wind_shear     # ref=HH=90, shear=0.12
dwind_profile(z) = shear · (1/HH)^shear · z^(shear-1)           # dudz factor
u_initial_sorted = ws · wind_profile(z_sorted)                  # ws=v∞ ; speed_ups=1
dudz_initial_sorted = ws · dwind_profile(z_sorted)
v_initial_sorted = 0 ;  w_initial_sorted = 0
u_sorted = u_initial.copy() ; v_sorted, w_sorted = 0
turbulence_intensity_field = 0.06 · ones(1,1,N,1,1)
```
`wind_veer = 0` so no veer rotation. `heterogenous_inflow_config = None` →
`speed_ups = 1.0`.

### 5.3 Per-turbine loop (sorted order `i = 0…N-1`)

For each turbine `i` (solver.py:91-255):

**Current-turbine scalars** (mean over rotor grid, then `[:,:,:,None,None]`):
`x_i, y_i, z_i = mean(grid.{x,y,z}_sorted[:,:,i:i+1], axis=(3,4))`.
`u_i = u_sorted[:,:,i:i+1]`, `v_i = v_sorted[:,:,i:i+1]` (full 3×3).

**Ct** (turbine.py:276-362), cubic-mean rotor average then table lookup:
```
ū = cubic_mean(u_sorted) = cbrt(mean(u^3, axis=(3,4)))      # "cubic-mean"
Ct_table = clip(fCt_interp(ū), 0.0001, 0.9999)              # interp1d, see §5.5
Ct_i = Ct_table · cosd(yaw) · cosd(tilt - 5.0)              # tilt term = 1
```
`fCt_interp` = `scipy.interp1d(wind_speed_tbl, thrust_tbl, bounds_error=False,
fill_value=(0.0001, 0.9999))` — linear, clamped to (0.0001,0.9999) outside range.
`Ct_i` is filtered to turbine `i` then `[:,:,0:1,None,None]`.

**Axial induction** (turbine.py:365-443):
```
a_i = 0.5 / (cosd(yaw)·cosd(tilt-5)) · (1 - sqrt(1 - Ct_i·cosd(yaw)·cosd(tilt-5)))
```
where `Ct_i` here is the *effective* Ct returned above (already ×cos yaw ×cos
tilt). Concretely with tilt inert: `a = 0.5/cosd(yaw)·(1 - sqrt(1 - Ct_eff))`,
`Ct_eff = Ct_table·cosd(yaw)`.

**effective_yaw_i** = `yaw_i` + (secondary steering `added_yaw`, below).

**Secondary steering — `wake_added_yaw`** (deflection/gauss.py:249-347,
`scale=1.0`):
```
D=126, HH=90, eps = 0.2·D
Uinf = mean(u_initial, axis=(2,3,4))                        # scalar freestream avg
vel_top    = ((HH + D/2)/HH)^shear ;  vel_bottom = ((HH - D/2)/HH)^shear
Γ_top    =  (π/8)·D·vel_top·Uinf·Ct_i
Γ_bottom = -(π/8)·D·vel_bottom·Uinf·Ct_i
ū3 = cbrt(mean(u_i^3, axis=(3,4)))
Γ_wake_rot = 0.25·2π·D·(a - a^2)·ū3 / TSR                   # TSR=8
yLocs = (y_sorted[i]-y_i) + NUM_EPS
zT = z_i-(HH+D/2)+NUM_EPS; rT = yLocs^2+zT^2; coreT = 1-exp(-rT/eps^2)
v_top    = mean( Γ_top·zT/(2π·rT)·coreT , axis=(3,4))
zB = z_i-(HH-D/2)+NUM_EPS; rB = yLocs^2+zB^2; coreB = 1-exp(-rB/eps^2)
v_bottom = mean( Γ_bottom·zB/(2π·rB)·coreB , axis=(3,4))
zC = z_i-HH+NUM_EPS; rC = yLocs^2+zC^2; coreC = 1-exp(-rC/eps^2)
v_core   = mean( Γ_wake_rot·zC/(2π·rC)·coreC , axis=(3,4))
avg_v = mean(v_i, axis=(3,4))
val = clip( 2·(avg_v - v_core)/(v_top + v_bottom), -1, 1)
added_yaw = degrees(0.5·arcsin(val))                        # [:,:,:,None,None]
```
Note `z_i` here is the current turbine's grid z-column `grid.z_sorted[:,:,i:i+1]`
(NOT the mean-scalar) — solver.py:155 passes `grid.z_sorted[:,:,i:i+1]`.

**Deflection field — `GaussVelocityDeflection.function`** (deflection/gauss.py:106-222):
```
yaw = -1·effective_yaw_i        # opposite sign convention; tilt=0
uR = U∞·Ct·cos(tilt)·cos(yaw) / (2·(1 - sqrt(1 - Ct·cos(tilt)·cos(yaw))))
u0 = U∞·sqrt(1 - Ct)
x0 = D·(cos(yaw)·(1+sqrt(1-Ct·cos(yaw)))) /
     (sqrt2·(4·alpha·TI + 2·beta·(1-sqrt(1-Ct)))) + x_i     # near/far boundary
ky = kz = ka·TI + kb                                        # ka=.38, kb=.004
C0 = 1 - u0/U∞ ;  M0 = C0·(2-C0)
E0 = C0^2 - 3·e^(1/12)·C0 + 3·e^(1/3)
sigma_z0 = D·0.5·sqrt(uR/(U∞+u0))
sigma_y0 = sigma_z0·cos(yaw)·cos(veer)                      # veer=0
xR = x_i
theta_c0 = dm·(0.3·radians(yaw)/cos(yaw))·(1 - sqrt(1-Ct·cos(yaw)))   # dm=1
delta0 = tan(theta_c0)·(x0 - x_i)
# near wake (mask xR ≤ x ≤ x0):
delta_near = ((x-xR)/(x0-xR))·delta0 + (ad + bd·(x-x_i))    # ad=bd=0
delta_near *= (x>=xR)&(x<=x0)
# far wake (mask x>x0):
sigma_y = ky·(x-x0)+sigma_y0  (x≥x0)  else sigma_y0
sigma_z = kz·(x-x0)+sigma_z0  (x≥x0)  else sigma_z0
ln_num = (1.6+√M0)·(1.6·√(σy·σz/(σy0·σz0)) - √M0)
ln_den = (1.6-√M0)·(1.6·√(σy·σz/(σy0·σz0)) + √M0)
delta_far = delta0 + theta_c0·E0/5.2·sqrt(σy0·σz0/(ky·kz·M0))·log(ln_num/ln_den)
            + (ad + bd·(x-x_i))
delta_far *= (x>x0)
deflection = delta_near + delta_far
```
`U∞` = `freestream_velocity = u_initial_sorted` (full field, broadcast). All
`cos/sin` in degrees except `theta_c0` uses `radians(yaw)` then `np.tan`.

**Transverse velocities — `calculate_transverse_velocity`**
(deflection/gauss.py:350-483, `scale=1.0`). Uses **unmodified** `yaw_angle_i`
(not effective), full grid `x/y/z_sorted − x_i / − y_i`:
```
D,HH,Ct,TSR,a as above ; eps = 0.2·D
Γ_top    =  sind(yaw)·cosd(yaw)·(π/8)·D·((HH+D/2)/HH)^shear·Uinf·Ct
Γ_bottom = -sind(yaw)·cosd(yaw)·(π/8)·D·((HH-D/2)/HH)^shear·Uinf·Ct
Γ_wake_rot = 0.25·2π·D·(a-a^2)·cbrt(mean(u_i^3,(3,4)))/TSR
# mixing-length decay:
lmda = D/8 ; kappa = 0.41
lm = kappa·z/(1 + kappa·z/lmda)
nu = lm^2·|dudz_initial|
decay = eps^2/(4·nu·delta_x/Uinf + eps^2)      # delta_x = x_sorted - x_i
yLocs = delta_y + NUM_EPS                       # delta_y = y_sorted - y_i
```
Six vortices (3 real at zT,zB,zC minus HH offsets + 3 ground-mirror at +HH
offsets), each contributing V,W via `Γ·z*/(2π·r*)·core·decay` and
`∓Γ·yLocs/(2π·r*)·core·decay`; see source lines 416-462 for exact zT/zB/zC/zTb/…
definitions (real: `z-(HH±D/2)`, `z-HH`; mirror: `z+(HH±D/2)`, `z+HH`, all +NUM_EPS;
mirror V,W carry an extra −1 / +1 sign, lines 447-462). Then:
```
V = V1+V2+V3+V4+V5+V6 ; W = W1+…+W6
V = where(delta_x >= 0, V, 0) ; W = where(delta_x >= 0, W, 0)
W = where(W >= 0, W, 0)                          # W clamped non-negative
```
Returns `(v_wake, w_wake)` of full grid shape. These **overwrite** `v_wake,
w_wake` each iteration (not accumulated within the loop; applied to
`flow_field.v/w_sorted +=` at loop end, so they accumulate across turbines via
the flow field).

**Yaw-added recovery — `yaw_added_turbulence_mixing`** (deflection/gauss.py:485-517),
only when enabled (it is):
```
I_i = turbulence_intensity_i[:,:,0,0,0]
ū3 = cbrt(mean(u_i^3, axis=(2,3,4)))
k  = (ū3·I_i)^2 / (2/3)
u_term = sqrt(2k)
v_term = mean(v_i + v_wake[:,:,i:i+1], axis=(2,3,4))
w_term = mean(w_i + w_wake[:,:,i:i+1], axis=(2,3,4))
k_total = 0.5·(u_term^2 + v_term^2 + w_term^2)
I_total = sqrt((2/3)·k_total)/ū3
I_mixing = I_total - I_i
turbine_turbulence_intensity[:,:,i] = turbulence_intensity_i + 2·I_mixing   # gch_gain=2
```

**Velocity deficit — `GaussVelocityDeficit.function`** (velocity/gauss.py:57-193):
```
yaw = -1·yaw_angle_i        # opposite sign; uses raw yaw not effective
uR = u_initial·Ct/(2·(1-sqrt(1-Ct)))
u0 = u_initial·sqrt(1-Ct)
sigma_z0 = D·0.5·sqrt(uR/(u_initial+u0))
sigma_y0 = sigma_z0·cos(yaw)·cos(veer)
xR = x_i
x0 = D·cos(yaw)·(1+sqrt(1-Ct)) / (sqrt2·(4·alpha·TI + 2·beta·(1-sqrt(1-Ct)))) + x_i
near_mask = (x > xR+0.1)&(x < x0) ; far_mask = (x >= x0)
# NEAR wake:
ramp_up = (x-xR)/(x0-xR) ; ramp_down = (x0-x)/(x0-xR)
sigma_y = ramp_down·0.501·D·sqrt(Ct/2) + ramp_up·sigma_y0     # ·(x>=xR), +0.5D·(x<xR)
sigma_z = (same with z)
# FAR wake:
ky=kz=ka·TI+kb ; sigma_y = ky·(x-x0)+sigma_y0 (far) else sigma_y0 ; sigma_z likewise
# r,C from rC(): (veer=0 ⇒ a=1/(2σy²), b=0, c=1/(2σz²))
r = (y - y_i - deflection)^2/(2σy²) + (z - HH)^2/(2σz²)
C = 1 - sqrt(clip(1 - Ct·cosd(yaw)/(8·σy·σz/D²), 0, 1))
deficit = C·exp(-r/(2·(√0.5)²)) = C·exp(-r)      # gaussian_function(C,r,1,√0.5)
velocity_deficit = near_deficit·near_mask + far_deficit·far_mask
```
`HH` in `rC` is `hub_height_i = 90`. `z` is grid z (see §5.1 open question).

**Combination — SOSFS** (wake_combination/sosfs.py:29-42):
```
wake_field = hypot(wake_field, velocity_deficit · u_initial_sorted)
           = sqrt(wake_field² + (velocity_deficit·u_initial)²)
```
Sqrt-sum-of-squares accumulation of absolute deficits (m/s), across turbines.

**Crespo-Hernandez added turbulence** (wake_turbulence/crespo_hernandez.py:68-98):
```
delta_x = x_sorted - x_i
down_mask = delta_x > -0.1 ; up_mask = delta_x <= 0.1
delta_x = delta_x·down_mask + 1·up_mask                       # avoid /0
ti = 0.5 · a_i^0.8 · ambient_TI^0.1 · (delta_x/D)^(-0.32)     # constant=.5, ai=.8, initial=.1
wake_added_TI = ti · down_mask
```
`ambient_TI = 0.06`. `a_i` = axial_induction_i.

**WAT integration** (solver.py:231-251):
```
area_overlap = sum(velocity_deficit·u_initial > 0.05, (3,4)) / (3·3)
ti_added = area_overlap · nan_to_num(wake_added_TI, posinf=0)
           · (x_sorted > x_i) · (|y_i - y_sorted| < 2·D) · (x_sorted <= 15·D + x_i)
turbine_turbulence_intensity = max( sqrt(ti_added² + ambient_TI²),
                                    turbine_turbulence_intensity )   # elementwise
```

**Apply to flow** (solver.py:253-255, end of iteration):
```
u_sorted = u_initial_sorted - wake_field       # note: from initial, not cumulative subtract
v_sorted += v_wake
w_sorted += w_wake
```
Because `wake_field` is the SOSFS accumulation, `u = u_initial − sqrt(Σ deficit²)`.
`u_sorted` is overwritten from `u_initial` each iteration (deficits combine in
`wake_field`, not in `u`). `v/w_sorted` accumulate `v_wake/w_wake` additively.

After the loop (solver.py:257-261):
```
turbulence_intensity_field_sorted = turbine_turbulence_intensity           # (1,1,N,1,1)
turbulence_intensity_field_sorted_avg = mean(…, (3,4))[:,:,:,None,None]
```

### 5.4 Finalize (unsort) — flow_field.py:229-241, floris.py:329-334

```
u = take_along_axis(u_sorted, unsorted_indices, 2)     # back to input turbine order
v, w likewise
turbulence_intensity_field = mean( take_along_axis(TI_field_sorted, unsorted, 2), (3,4))
                             # shape (1,1,N)  — per-turbine scalar TI
farm.yaw_angles = unsort(yaw_angles_sorted)
```

### 5.5 Interpolation details (must replicate exactly)

All three are `scipy.interpolate.interp1d(kind="linear", bounds_error=False)`:
- `fCt_interp`: `fill_value=(0.0001, 0.9999)` (below-range → 0.0001, above → 0.9999),
  then result additionally `clip(·, 0.0001, 0.9999)`.
- `fCp_interp`: `fill_value=(0.0, 1.0)`.
- `power_interp`: interpolates **pre-computed inner power**
  `inner_power = 0.5·rotor_area·fCp_interp(ws)·gen_eff·ws³` (turbine.py:663-674),
  `fill_value=0`, `bounds_error=False`. `rotor_area = π·(D/2)² = π·63²`.

`interp1d` linear with tuple `fill_value` = constant hold outside each end
(NOT extrapolation). Duplicate wind-speed rows are removed via
`np.unique(return_index=True)` before building interps (turbine.py:539-543) —
the nrel_5MW table has no dups, so order is preserved.

---

## 6. Outputs

### 6a. Power (turbine.py `power`, `rotor_effective_velocity`)

```
ū = cubic_mean(u, axis=(3,4)) = cbrt(mean(u³))                    # unsorted flow
U_eff = (air_density/ref_density_cp_ct)^(1/3) · ū                # (1.225/1.225)^… = 1
U_eff = U_eff · cosd(yaw)^(pP/3)                                  # pP=1.88 → exponent .6267
U_eff = U_eff · cosd(tilt - 5)^(pT/3)                             # = 1
P = power_interp(U_eff) · ref_density_cp_ct                       # ·1.225, Watts
```
`power_interp(U_eff)` returns `0.5·rotor_area·Cp(U_eff)·gen_eff·U_eff³` per §5.5.
So `P = 0.5·1.225·π·63²·Cp(U_eff)·U_eff³` W. WFCRL `avg_powers` flattens this
to `(N,)` float32 (interface.py:621-622); mdp divides by `1e6` → **MW**
(mdp.py:281).

### 6b. Loads proxy (interface.py `local_load_proxies`, 628-636)

```
turbulences = turbulence_intensity_field.squeeze()               # (N,) per-turbine TI
var_u = std(flow_field.u, axis=(3,4)).squeeze()                  # (N,) rotor std of u
var_v = std(flow_field.v, axis=(3,4)).squeeze()
var_w = std(flow_field.w, axis=(3,4)).squeeze()
loads_raw = stack([turbulences, var_u, var_v, var_w]).T          # (N,4)
current_measures[:,[3,4,5,6]] = loads_raw · 1e7                  # stored ×1e7
```
`np.std` is population std (ddof=0). `get_measure("load")` → `(N,4)`; mdp then
`loads /= 1e7` (mdp.py:280) → back to `(N,4)` raw. `load_penalty =
mean(|loads|)` over all `4N` entries.

### 6c. Local wind measurement (interface.py `local_wind_measurements`, 638-647)

```
speed_local = cbrt(mean(flow_field.u³, axis=(3,4))).squeeze()    # (N,) ∛(mean(u³))
dir_local   = mean( wind_dir_free - degrees(arctan2(v, u)), axis=(3,4)).squeeze()
```
`wind_dir_free = fi.floris.flow_field.wind_directions[0]`.

### 6d. Observation dict layout

`state_attributes = list(controls) + measures`. With `controls={"yaw":…}` and
`FlorisInterface.measure_map = {yaw, wind_speed, wind_direction, load,
freewind_measurements}`, `measures = [freewind_measurements, wind_speed,
wind_direction]` (yaw excluded as control; pitch/torque/load excluded — not both
in `POSSIBLE_STATE_ATTRIBUTES` and floris measure_map). **`load` is never in the
observation** — reward only.

**WindFarmEnv (central)** observation — ordered dict:
| key | shape | value |
|---|---|---|
| `yaw` | `(N,)` | integrated absolute yaw (deg), = `state["yaw"]` (NOT read back from FLORIS) |
| `freewind_measurements` | `(2,)` | `[wind_speed_free, wind_dir_free]` |
| `wind_speed` | `(N,)` | local `∛(mean u³)` per turbine |
| `wind_direction` | `(N,)` | local direction per turbine |

Box dtype float64, clipped to `state_space` bounds (`yaw`∈[-40,40], `wind_speed`
∈[0,28], `wind_direction`∈[0,360], `freewind` [0,28]×[0,360]).

**MAWindFarmEnv** per-agent obs (multiagent_env.py:102-115): same keys **minus**
`freewind_measurements` (in `IGNORE_GLOBAL_ATTRIBUTES`), each sliced to that
agent's index → scalars: `{yaw: yaw[i], wind_speed: ws[i], wind_direction:
wd[i]}`. Agents = `turbine_1…turbine_N`, AEC turn order = that list;
reward/state advance only when `_agent_selector.is_last()` (all agents acted).

---

## 7. Reward

Central (simple_env.py:74-85) and MA (multiagent_env.py:216-227) identical:
```
v = state_before_step["freewind_measurements"][0]                # freestream speed
normalized_powers = powers_MW · 1e3 / v³                         # per turbine, = kW/v³
load_penalty = mean(|loads|)          if loads is not None else 0   # loads (N,4), raw
reward = mean(normalized_powers) - load_coef · load_penalty      # load_coef = 0.1 default
reward = reward_shaper(reward)         # DoNothingReward = identity by default
```
`powers_MW = get_turbine_powers()/1e6`. `v` is the **freestream** speed from the
state at step entry (the episode-constant `wind_speed_free`), not a per-turbine
local value. Scalar reward, broadcast to all agents in MA.

---

## 8. Episode / termination

Fixed-horizon truncation only. `terminated` always `False`. `truncated` = the
FLORIS interface's `_num_iter == max_iter` flag (interface.py:585), where
`max_iter = horizon = start_iter + max_num_steps` (mdp.py:33,68). `max_num_steps`
default 500. No early termination, no reward-based done.

---

## 9. Numerical-agreement notes

- **Precision**: FLORIS `floris_float_type = np.float64`; the entire wake solve
  is float64. WFCRL rounds state/powers through float32 at the env boundary
  (§2). JAX defaults to float32 — **enable `jax.config.update("jax_enable_x64",
  True)`** for the solve. Target tolerance against the reference: with x64,
  expect `< 1e-8` relative on powers; with float32 the Gaussian exponentials and
  the SOSFS accumulation drift to `~1e-3`–`1e-2` relative — not acceptable for a
  numerical-agreement test. Recommend x64 for the solve, cast to float32 only at
  the observation boundary if reproducing WFCRL's rounding exactly.
- **argsort ordering**: the turbine loop order is `_x.argsort(axis=2)`, a fixed
  permutation **per wind direction**. In JAX, precompute the permutation from
  the rotated x-coordinates per `(env, wd)` and gather; it is static per
  condition (does not depend on yaw or the flow solve). Use a stable sort if you
  need bitwise tie-breaking parity; real layouts have no x-ties.
- **Boolean masks → `where/select`**: FLORIS multiplies by boolean arrays
  (`·(x>=xR)`, near/far masks, WAT gates, `np.where` in transverse velocity).
  Port these as `jnp.where`/multiply — but note the *branch guards*
  `if np.sum(near_wake_mask):` (velocity/gauss.py:124,166) skip whole-array
  computation when empty. In JAX (no data-dependent control flow under jit)
  always compute both branches and mask; results are identical because the mask
  zeroes the skipped region. Same for the `if not np.all(farm.yaw_angles):`
  turbopark path — not on the gauss path, ignore.
- **`interp1d` edge behavior**: constant-hold outside range with the exact
  `fill_value` tuples in §5.5, plus the post-clip on Ct. Implement as
  `jnp.interp` (which clamps to endpoints) then override with the fill constants
  where `ws` is out of `[tbl_min, tbl_max]`, then clip Ct to (0.0001,0.9999).
  `jnp.interp` endpoint-clamp ≠ scipy's tuple `fill_value` when the fill differs
  from the endpoint (Ct table ends at 0.0/0.0 vs fill 0.0001/0.9999 — must
  override explicitly).
- **`nan_to_num`**: WAT uses `nan_to_num(wake_added_TI, posinf=0.0)` — replicate
  (Crespo `(delta_x/D)^-0.32` → inf at `delta_x→0`, masked out anyway, but
  `nan_to_num` must run before the mask multiply to match).
- **`numexpr`**: FLORIS evaluates several expressions via `ne.evaluate` (float64
  throughout) — mathematically identical to numpy; no special handling.
- **NUM_EPS = 0.001**: added inside every GCH `r*`/`yLocs`/`z*` to avoid /0 —
  copy exactly; it perturbs results at the 1e-3 level near singularities.
- **`np.std` ddof=0** for the loads proxy.
- **z-grid / hub-height convention** (§5.1): resolved — grid z is centered on
  HH=90 (`Vec3([x,y,hub_height])`), so `z_grid = 90 + [-31.5,0,31.5]`. No
  ambiguity; noted here only because it is easy to get wrong (the rotor plane is
  NOT centered on the tower base z=0).

---

## 10. Deviations the JAX port may take (result-preserving)

1. **No per-step object rebuild.** WFCRL/FLORIS rebuild grid/farm/flow objects
   every `calculate_wake`; the port precomputes static geometry once and only
   re-runs the numeric solve. Wind is constant for HornsRev2 so `reinitialize`
   never fires after reset.
2. **Precomputed sort permutation** per wind direction (§9) instead of argsort
   inside the step.
3. **Batched conditions / envs** on the `(n_wd, n_ws)` and an added `n_env` axis
   via `vmap`; the reference loops one condition at a time.
4. **Skip tilt entirely** (inert, §5.0) — drop `tilt`, `ref_tilt`, all
   `cosd(tilt-5)=1` factors and the floating-tilt interpolation.
5. **Skip heterogeneous inflow / veer** (`het_map=None`, `veer=0`) — `speed_ups=1`,
   veer rotation is identity.
6. **Compute both near/far branches unconditionally and mask** instead of the
   `if np.sum(mask):` guards (§9) — identical numerics, jit-compatible.
7. **Keep float64 end-to-end** rather than reproducing WFCRL's float32 boundary
   rounding — closer to the true physics; only reintroduce float32 casts if a
   test demands bit-parity with the WFCRL observation, not the FLORIS core.
8. **Drop the unused solver branches** (cc / turbopark / empirical / multidim)
   and the full-flow-field visualization solvers — only `sequential_solver` runs
   for the GCH velocity model `gauss`.
9. **Omit `pitch`/`torque` controls and the MPI/FastFarm interfaces** — the
   FLORIS interface only exposes `yaw`.
10. **Optional `fidelity="corrected"` flag** (static; default `"floris"` is
    bit-identical) removing three reference quirks stable across FLORIS 3.5/4.6.6:
    deterministic rotor-plane self-exclusion (vs the `x_i` mean-rounding ULP gate),
    consistent yaw-added-recovery TI ordering (deflection sees the updated TI, not
    stale), and discrete duty-limited turbines holding (vs raw-zeroing mapping
    `0 -> -step`). *(Added; owner review.)*
11. **Optional `turbine="nrel5mw_v4"` library** (default `"nrel5mw_v3"` unchanged):
    FLORIS 4.6.6 NREL-5MW (rotor_diameter 125.88, cosine-loss absolute-kW power
    table) threaded as a `TurbineSpec` PyTree through the physics surface.
    *(Added; owner review.)*
