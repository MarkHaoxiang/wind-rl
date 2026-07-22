"""env-layer behavior derivable from spec §4/§7/§8 without the wake solve."""

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
)
from windrl_engine.farm.layout import row_layout


def test_action_pipeline_constants_match_spec_section_4a() -> None:
    # simple_env.py ACTUATORS_RATE["yaw"]=0.3 deg/s, FlorisInterface.dt=60s,
    # over-active threshold 0.1, DEFAULT_BOUNDS["yaw"]=[-40,40] (spec §2, §4a).
    assert SLEW_RATE == 0.3
    assert DT == 60.0
    assert DUTY_FRACTION == 0.1
    assert YAW_LIMIT == 40.0


def test_duty_cycle_limiter_passes_action_through_on_the_first_step() -> None:
    # step_count IS num_moves directly (reset's state.step_count starts at 1,
    # §3/§8 burn-in convention), so the first real agent step passes step_count=1.
    action = jnp.asarray([5.0, -5.0])
    accumulator = jnp.zeros(2)
    out = duty_cycle_limiter(action, accumulator, step_count=jnp.asarray(1))
    assert jnp.array_equal(out, action)


def test_duty_cycle_limiter_zeros_action_once_accumulator_hits_10_percent_duty() -> (
    None
):
    # Hand-derived from spec §4a: actuating_frac = accumulator/0.3/num_moves/60.
    # After one full-rate step of 5 deg (num_moves=1), accumulator=5 carries
    # into the *second* call (num_moves=2): 5/0.3/2/60 = 0.1389 >= 0.1 -> zeroed.
    accumulator = jnp.asarray([5.0, 5.0])
    action = jnp.asarray([5.0, -5.0])
    out = duty_cycle_limiter(action, accumulator, step_count=jnp.asarray(2))
    assert jnp.array_equal(out, jnp.zeros(2))


def test_duty_cycle_limiter_boundary_is_inclusive_at_exactly_10_percent() -> None:
    # actuating_frac == 0.1 exactly must zero (spec: ">= 0.1"), not "> 0.1".
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
    # Same slewing-max-Delta=5-every-step scenario as spec §4a, run through the
    # full pipeline: yaw should move on calls 1 and 3, and hold on call 2 (the
    # first zeroed step) and call 4. step_count IS num_moves (1-indexed, per the
    # reset burn-in convention), so call i (0-indexed) passes step_count=i+1.
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


def test_reset_initializes_step_count_to_1_for_the_burn_in_solve() -> None:
    # Spec §3: reset runs a 1-step zero-yaw burn-in before the first agent
    # action (mirroring WFCRL's `_num_iter` convention), so the fresh state's
    # step_count starts at 1, not 0.
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
            layout, state, jnp.zeros(2), yaw_step=5.0, load_coef=0.1, horizon=horizon
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
