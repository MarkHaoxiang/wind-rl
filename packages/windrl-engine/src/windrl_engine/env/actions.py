from typing import Final, Literal, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

ControlMode = Literal["continuous", "discrete"]
Fidelity = Literal["floris", "corrected"]

SLEW_RATE: Final = 0.3  # deg/s, ACTUATORS_RATE["yaw"] (simple_env.py)
DT: Final = 60.0  # s, FlorisInterface.dt (interface.py)
DUTY_FRACTION: Final = 0.1  # over-active threshold (simple_env.py)
YAW_LIMIT: Final = 40.0  # deg, DEFAULT_BOUNDS["yaw"] (mdp.py)


def duty_over_active(
    accumulator: Float[Array, "turbines"],
    step_count: Int[Array, ""],
) -> Float[Array, "turbines"]:
    """Boolean mask: turbines slewing over ``DUTY_FRACTION`` of elapsed wall time (spec §4a).

    WFCRL's ``num_moves`` counts agent steps only (simple_env.py:64, independent of the
    reset burn-in), so it equals ``step_count`` entering the step now that the
    reset-produced state starts at 1 (§8); the accumulator holds ``Σ|applied Δyaw|``.
    """
    num_moves = step_count
    actuating_frac = accumulator / SLEW_RATE / num_moves / DT
    return actuating_frac >= DUTY_FRACTION


def duty_cycle_limiter(
    action: Float[Array, "turbines"],
    accumulator: Float[Array, "turbines"],
    step_count: Int[Array, ""],
) -> Float[Array, "turbines"]:
    """Zero the raw action of over-active turbines (floris fidelity, spec §4a).

    Applied to the raw action *before* the discrete mapping, so a zeroed discrete
    action later maps to ``-step`` (a down move) — the FLORIS reference quirk.
    """
    return jnp.where(duty_over_active(accumulator, step_count), 0.0, action)


def command_from_action(
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    control_mode: ControlMode,
) -> Float[Array, "turbines"]:
    """Map a raw action to a Δyaw command (deg), spec §4b.

    Continuous: clip to the action box ``[-yaw_step, yaw_step]``.
    Discrete: ``{0, 1, 2} -> {-yaw_step, 0, +yaw_step}``.
    """
    if control_mode == "continuous":
        return jnp.clip(action, -yaw_step, yaw_step)
    return (action - 1.0) * yaw_step


class AppliedAction(NamedTuple):
    yaw: Float[Array, "turbines"]  # absolute deg, clipped to [-40, 40]
    accumulator: Float[Array, "turbines"]  # Σ|applied Δyaw| deg


def apply_action(
    yaw: Float[Array, "turbines"],
    accumulator: Float[Array, "turbines"],
    step_count: Int[Array, ""],
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    control_mode: ControlMode,
    fidelity: Fidelity = "floris",
) -> AppliedAction:
    """Full single-farm action pipeline (spec §4): limiter, map, integrate.

    The accumulator grows by the post-limiter, post-box-clip command magnitude
    (not the ±40-saturated yaw change), matching mdp.py:288-316.

    ``fidelity="corrected"``: the duty limiter zeros the *mapped* command, so a
    duty-limited discrete turbine holds (command 0) instead of the reference's
    raw-zeroing that maps ``0 -> -step`` (a down move). Continuous control is
    unaffected (clipping a zeroed raw action also yields command 0).
    """
    if fidelity == "corrected":
        command = command_from_action(
            action, yaw_step=yaw_step, control_mode=control_mode
        )
        command = jnp.where(duty_over_active(accumulator, step_count), 0.0, command)
    else:
        limited = duty_cycle_limiter(action, accumulator, step_count)
        command = command_from_action(
            limited, yaw_step=yaw_step, control_mode=control_mode
        )
    yaw_abs = jnp.clip(yaw + command, -YAW_LIMIT, YAW_LIMIT)
    return AppliedAction(yaw=yaw_abs, accumulator=accumulator + jnp.abs(command))
