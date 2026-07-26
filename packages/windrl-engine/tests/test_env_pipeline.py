"""Env action-pipeline behavior (duty limiter, yaw integration, truncation) without the wake solve."""

import jax
import jax.numpy as jnp

from windrl_engine.env.actions import (
    DT,
    DUTY_FRACTION,
    SLEW_RATE,
    YAW_LIMIT,
    apply_action,
    command_from_action,
    duty_cycle_limiter,
)
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import (
    WIND_DIRECTION_MAX,
    WIND_SPEED_MAX,
    BatchedWindFarmEnv,
    reset,
    step,
    wfcrl_reward,
)
from windrl_engine.farm.layout import ablaincourt, row_layout, turb3_row1
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import load_proxies, turbine_powers
from windrl_engine.physics.solver import solve_farm


def test_action_pipeline_constants_match_wfcrl_defaults() -> None:
    # Matches WFCRL's yaw actuator rate (0.3 deg/s), FLORIS interface timestep
    # (60s), duty-cycle threshold (0.1), and yaw bound ([-40, 40]).
    assert SLEW_RATE == 0.3
    assert DT == 60.0
    assert DUTY_FRACTION == 0.1
    assert YAW_LIMIT == 40.0


def test_duty_cycle_limiter_passes_action_through_on_the_first_step() -> None:
    # step_count IS num_moves directly (reset's state.step_count starts at 1,
    # matching WFCRL's burn-in convention), so the first real agent step passes
    # step_count=1.
    action = jnp.asarray([5.0, -5.0])
    accumulator = jnp.zeros(2)
    out = duty_cycle_limiter(action, accumulator, step_count=jnp.asarray(1))
    assert jnp.array_equal(out, action)


def test_duty_cycle_limiter_zeros_action_once_accumulator_hits_10_percent_duty() -> (
    None
):
    # actuating_frac = accumulator/0.3/num_moves/60. After one full-rate step of
    # 5 deg (num_moves=1), accumulator=5 carries into the *second* call
    # (num_moves=2): 5/0.3/2/60 = 0.1389 >= 0.1 -> zeroed.
    accumulator = jnp.asarray([5.0, 5.0])
    action = jnp.asarray([5.0, -5.0])
    out = duty_cycle_limiter(action, accumulator, step_count=jnp.asarray(2))
    assert jnp.array_equal(out, jnp.zeros(2))


def test_duty_cycle_limiter_boundary_is_inclusive_at_exactly_10_percent() -> None:
    # actuating_frac == 0.1 exactly must zero (the check is ">= 0.1", not "> 0.1").
    # num_moves=5 is chosen so accumulator/0.3/num_moves/60 round-trips to
    # bit-exact 0.1 in float64 (decimal fractions like 0.1/0.3 don't divide
    # exactly in binary for arbitrary num_moves).
    num_moves = 5
    accumulator = jnp.asarray([DUTY_FRACTION * SLEW_RATE * num_moves * DT])
    assert float(accumulator[0] / SLEW_RATE / num_moves / DT) == 0.1
    out = duty_cycle_limiter(
        jnp.asarray([5.0]), accumulator, step_count=jnp.asarray(num_moves)
    )
    assert jnp.array_equal(out, jnp.zeros(1))


def test_command_from_action_continuous_clips_to_the_yaw_step_box() -> None:
    action = jnp.asarray([7.0, -8.0, 3.0])
    command = command_from_action(action, yaw_step=5.0, control_mode="continuous")
    assert jnp.array_equal(command, jnp.asarray([5.0, -5.0, 3.0]))


def test_command_from_action_discrete_maps_0_1_2_to_minus_0_plus_step() -> None:
    action = jnp.asarray([0.0, 1.0, 2.0])
    command = command_from_action(action, yaw_step=5.0, control_mode="discrete")
    assert jnp.array_equal(command, jnp.asarray([-5.0, 0.0, 5.0]))


def test_apply_action_clips_absolute_yaw_to_plus_minus_40() -> None:
    result = apply_action(
        yaw=jnp.asarray([38.0, -38.0]),
        accumulator=jnp.zeros(2),
        step_count=jnp.asarray(1),
        action=jnp.asarray([10.0, -10.0]),
        yaw_step=5.0,
        control_mode="continuous",
    )
    assert jnp.array_equal(result.yaw, jnp.asarray([40.0, -40.0]))
    assert jnp.array_equal(result.accumulator, jnp.asarray([5.0, 5.0]))


def test_apply_action_duty_cycle_zeroing_first_bites_on_the_second_call() -> None:
    # Same slewing-max-delta-5-every-step scenario, run through the full
    # pipeline: yaw should move on calls 1 and 3, and hold on call 2 (the first
    # zeroed step) and call 4. step_count IS num_moves (1-indexed, per the reset
    # burn-in convention), so call i (0-indexed) passes step_count=i+1.
    yaw = jnp.zeros(1)
    accumulator = jnp.zeros(1)
    expected_yaw_after_each_call = [5.0, 5.0, 10.0, 10.0]
    for i, expected in enumerate(expected_yaw_after_each_call):
        result = apply_action(
            yaw=yaw,
            accumulator=accumulator,
            step_count=jnp.asarray(i + 1),
            action=jnp.asarray([5.0]),
            yaw_step=5.0,
            control_mode="continuous",
        )
        yaw, accumulator = result.yaw, result.accumulator
        assert float(yaw[0]) == expected


def test_corrected_fidelity_holds_where_floris_fidelity_moves_down_on_a_duty_limited_discrete_up_action() -> (
    None
):
    # Duty-limited turbine picks discrete index 2 ("up"). "floris" fidelity zeros
    # the *raw action* first, and a zeroed discrete index maps to -yaw_step (a
    # down move); "corrected" fidelity zeros the *mapped command* instead, so the
    # same over-active turbine holds at a 0 delta.
    accumulator = jnp.asarray([5.0])
    step_count = jnp.asarray(2)  # 5/0.3/2/60 = 0.1389 >= DUTY_FRACTION
    action = jnp.asarray([2.0])

    floris = apply_action(
        yaw=jnp.zeros(1),
        accumulator=accumulator,
        step_count=step_count,
        action=action,
        yaw_step=5.0,
        control_mode="discrete",
        fidelity="floris",
    )
    corrected = apply_action(
        yaw=jnp.zeros(1),
        accumulator=accumulator,
        step_count=step_count,
        action=action,
        yaw_step=5.0,
        control_mode="discrete",
        fidelity="corrected",
    )
    assert float(floris.yaw[0]) == -5.0
    assert float(corrected.yaw[0]) == 0.0


def test_reset_initializes_step_count_to_1_for_the_burn_in_solve() -> None:
    # reset runs a 1-step zero-yaw burn-in before the first agent action
    # (matching WFCRL's convention), so the fresh state's step_count starts at
    # 1, not 0.
    layout = row_layout(2)
    state, _ = reset(layout, jax.random.key(0))
    assert int(state.step_count) == 1


def test_step_truncates_on_the_horizon_minus_1th_agent_step() -> None:
    # step_count starts at 1 (burn-in) and increments once per agent step;
    # truncation fires when step_count == horizon. horizon=5 therefore allows
    # exactly 4 agent steps, with `truncated` first True on the 4th call.
    layout = row_layout(2)
    state, _ = reset(layout, jax.random.key(0))
    horizon = 5
    expected_truncated = [False, False, False, True]
    for expected in expected_truncated:
        state, _, _, truncated = step(
            layout,
            state,
            jnp.zeros(2),
            yaw_step=5.0,
            reward_fn=wfcrl_reward(0.1),
            horizon=horizon,
        )
        assert bool(truncated) == expected
    assert int(state.step_count) == horizon


def test_observation_space_freewind_bounds_match_the_actual_clipping() -> None:
    # _observation clips freewind to [clip(speed, 0, WIND_SPEED_MAX),
    # clip(direction, 0, WIND_DIRECTION_MAX)] -- per-element bounds, not a
    # single (low, high) pair broadcast identically over both components.
    config = WindFarmEnvConfig(layout=[(0.0, 0.0), (504.0, 0.0)])
    env = BatchedWindFarmEnv(config)
    space = env.observation_space()["freewind"]

    low = jnp.broadcast_to(jnp.asarray(space.low, dtype=jnp.float64), space.shape)
    high = jnp.broadcast_to(jnp.asarray(space.high, dtype=jnp.float64), space.shape)
    assert jnp.allclose(low, jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(high, jnp.asarray([WIND_SPEED_MAX, WIND_DIRECTION_MAX]))


def test_build_layout_resolves_a_named_layout_string_to_its_builder() -> None:
    layout = WindFarmEnvConfig(layout="ablaincourt").build_layout()
    expected = ablaincourt()
    assert jnp.array_equal(layout.x, expected.x)
    assert jnp.array_equal(layout.y, expected.y)


def test_wfcrl_reward_is_mean_kw_per_cubed_freestream_minus_the_load_penalty() -> None:
    # Pinned against a hand calculation on a real solve, because the reward's
    # units are not recoverable from the code: powers arrive in W and are scaled
    # by 1e-6 then 1e3, i.e. kW, before the /speed^3 normalization. Both terms
    # must be checked -- the load penalty is a flat mean over all four proxies
    # (TI and the three velocity stds), not a per-turbine norm.
    layout = turb3_row1()
    wind = WindCondition(speed=jnp.asarray(9.0), direction=jnp.asarray(270.0))
    yaw = jnp.asarray([10.0, -5.0, 0.0])

    solution = solve_farm(layout, wind, yaw)
    powers = turbine_powers(solution.u, yaw)
    loads = load_proxies(solution)

    unpenalized = jnp.mean(powers / 1e3) / wind.speed**3
    assert float(wfcrl_reward(0.0)(powers, loads, wind.speed)) == float(unpenalized)

    penalized = unpenalized - 0.1 * jnp.mean(jnp.abs(loads))
    assert float(wfcrl_reward(0.1)(powers, loads, wind.speed)) == float(penalized)
