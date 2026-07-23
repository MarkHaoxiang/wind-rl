"""physics/ invariant tests, CI-safe (no wfcrl)."""

import jax
import jax.numpy as jnp

from windrl_engine.env.env import reset, step
from windrl_engine.farm.layout import FarmLayout, row_layout
from windrl_engine.farm.turbine import HUB_HEIGHT, D
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

# Rotor grid z-offsets (spec §5.1): disc_grid = linspace(-31.5, 31.5, 3), and
# the grid is centered on hub height (not the tower base) -- z_grid = HH + disc.
_GRID_Z = HUB_HEIGHT + jnp.asarray([-0.25 * D, 0.0, 0.25 * D])
RATED_POWER_W = 5.1e6


def _shear_ceiling(speed: jax.Array) -> jax.Array:
    """Freestream shear profile u(z) = ws·(z/HH)^0.12 (spec §5.2), per grid row."""
    return speed * (_GRID_Z / HUB_HEIGHT) ** 0.12


def test_solve_farm_is_equivariant_to_turbine_permutation() -> None:
    layout = row_layout(4)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.asarray([5.0, -3.0, 0.0, 8.0])
    perm = jnp.asarray([2, 0, 3, 1])

    baseline = solve_farm(layout, wind, yaw)
    permuted_layout = FarmLayout(x=layout.x[perm], y=layout.y[perm])
    permuted = solve_farm(permuted_layout, wind, yaw[perm])

    assert jnp.allclose(permuted.u, baseline.u[perm], atol=1e-9)
    assert jnp.allclose(permuted.v, baseline.v[perm], atol=1e-9)
    assert jnp.allclose(permuted.w, baseline.w[perm], atol=1e-9)
    assert jnp.allclose(
        permuted.turbulence_intensity, baseline.turbulence_intensity[perm], atol=1e-9
    )


def test_solve_farm_is_rotation_invariant_with_matched_wind_direction() -> None:
    layout = row_layout(4)
    yaw = jnp.asarray([0.0, 4.0, -6.0, 2.0])
    wind = WindCondition(speed=jnp.asarray(10.0), direction=jnp.asarray(270.0))

    theta = 37.0
    x_c = (layout.x.min() + layout.x.max()) / 2.0
    y_c = (layout.y.min() + layout.y.max()) / 2.0
    dx, dy = layout.x - x_c, layout.y - y_c
    rad = jnp.deg2rad(theta)
    rotated_layout = FarmLayout(
        x=dx * jnp.cos(rad) - dy * jnp.sin(rad) + x_c,
        y=dx * jnp.sin(rad) + dy * jnp.cos(rad) + y_c,
    )
    # Meteorological from-azimuth is clockwise-positive; the planar (x,y)
    # rotation above is counterclockwise-positive. A CCW layout rotation by
    # +theta must therefore pair with a CW wind-direction rotation, i.e.
    # subtract theta from the from-azimuth to keep the flow geometry matched.
    rotated_wind = WindCondition(
        speed=wind.speed, direction=(wind.direction - theta) % 360.0
    )

    baseline = solve_farm(layout, wind, yaw)
    rotated = solve_farm(rotated_layout, rotated_wind, yaw)

    baseline_power = turbine_powers(baseline.u, yaw)
    rotated_power = turbine_powers(rotated.u, yaw)
    assert jnp.allclose(rotated_power, baseline_power, atol=1e-9, rtol=1e-9)


def test_solve_farm_vmapped_over_conditions_matches_individual_solves() -> None:
    layout = row_layout(3)
    yaw = jnp.zeros(3)
    speeds = jnp.asarray([6.0, 9.0, 12.0])
    directions = jnp.asarray([260.0, 270.0, 280.0])
    conditions = WindCondition(speed=speeds, direction=directions)

    batched = jax.vmap(solve_farm, in_axes=(None, 0, None))(layout, conditions, yaw)

    for i in range(3):
        single = solve_farm(
            layout, WindCondition(speed=speeds[i], direction=directions[i]), yaw
        )
        assert jnp.allclose(batched.u[i], single.u, atol=1e-9)
        assert jnp.allclose(batched.v[i], single.v, atol=1e-9)
        assert jnp.allclose(batched.w[i], single.w, atol=1e-9)
        assert jnp.allclose(
            batched.turbulence_intensity[i], single.turbulence_intensity, atol=1e-9
        )


def test_solve_farm_deficit_never_exceeds_freestream_shear_and_stays_positive() -> None:
    layout = row_layout(4)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(4)

    solution = solve_farm(layout, wind, yaw)
    ceiling = _shear_ceiling(
        wind.speed
    )  # shape (grid,), broadcasts over (turbines, grid)

    assert bool(jnp.all(solution.u > 0.0))
    assert bool(jnp.all(solution.u <= ceiling[None, None, :] + 1e-9))


def test_solve_farm_downstream_turbine_is_strictly_waked_by_aligned_upstream() -> None:
    # 270 deg = wind from the west, flowing in +x; row_layout is aligned along
    # +x with y=0, so turbine 1 sits directly downstream of turbine 0.
    layout = row_layout(2)
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(2)

    solution = solve_farm(layout, wind, yaw)
    ceiling = _shear_ceiling(wind.speed)
    downstream_u = solution.u[1]

    assert bool(jnp.all(downstream_u < ceiling[None, :] - 1e-6))


def test_turbine_powers_never_exceed_rated_and_front_turbine_dominates_the_row() -> (
    None
):
    # GCH wake-added turbulence mixing lets deep-row turbines partially recover
    # (row power need not be monotone non-increasing), so the sharp invariants
    # here are: nothing exceeds the rated ceiling, the unwaked front turbine is
    # the row's strict maximum, and turbine 2 -- compounded (SOSFS) deficit
    # from both upstream wakes, before turbulence-mixing recovery has built up
    # -- is the row's strict minimum (recovery is then visible at turbines 3-4).
    layout = row_layout(5)
    wind = WindCondition(speed=jnp.asarray(11.0), direction=jnp.asarray(270.0))
    yaw = jnp.zeros(5)

    solution = solve_farm(layout, wind, yaw)
    powers = turbine_powers(solution.u, yaw)

    assert bool(jnp.all(powers <= RATED_POWER_W))
    assert bool(jnp.all(powers[0] > powers[1:]))  # front turbine: strict row max
    assert bool(
        jnp.all(powers[2] < powers[jnp.asarray([0, 1, 3, 4])])
    )  # strict row min


def _run_trajectory(
    layout: FarmLayout, key: jax.Array, actions: jax.Array
) -> list[jax.Array]:
    state, obs = reset(layout, key)
    trace = [obs.yaw, obs.wind_speed, obs.wind_direction]
    for action in actions:
        state, obs, reward, _truncated = step(
            layout, state, action, yaw_step=5.0, load_coef=0.1, horizon=10
        )
        trace.extend([obs.yaw, obs.wind_speed, obs.wind_direction, reward])
    return trace


def test_replay_from_same_key_and_action_stream_is_bitwise_deterministic() -> None:
    layout = row_layout(3)
    key = jax.random.key(42)
    actions = jnp.asarray([[5.0, -5.0, 0.0], [0.0, 5.0, -5.0], [-5.0, 0.0, 5.0]])

    first = _run_trajectory(layout, key, actions)
    second = _run_trajectory(layout, key, actions)

    for a, b in zip(first, second, strict=True):
        assert jnp.array_equal(a, b)
