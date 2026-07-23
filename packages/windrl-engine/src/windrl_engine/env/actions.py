from typing import Final, Literal, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

ControlMode = Literal["continuous", "discrete"]
Fidelity = Literal["floris", "corrected"]

SLEW_RATE: Final = 0.3  # deg/s, matches WFCRL's yaw actuator rate
DT: Final = 60.0  # s, matches WFCRL's FLORIS interface timestep
DUTY_FRACTION: Final = 0.1  # over-active threshold, matches WFCRL's default
YAW_LIMIT: Final = 40.0  # deg, matches WFCRL's yaw bound


def duty_over_active(
    accumulator: Float[Array, "turbines"],
    step_count: Int[Array, ""],
) -> Float[Array, "turbines"]:
    """Boolean mask: turbines slewing over ``DUTY_FRACTION`` of elapsed wall time."""
    # step_count counts agent steps starting from reset's step_count=1, matching
    # WFCRL's num_moves convention (independent of the reset burn-in); the
    # accumulator holds Σ|applied Δyaw|.
    num_moves = step_count
    actuating_frac = accumulator / SLEW_RATE / num_moves / DT
    return actuating_frac >= DUTY_FRACTION


def duty_cycle_limiter(
    action: Float[Array, "turbines"],
    accumulator: Float[Array, "turbines"],
    step_count: Int[Array, ""],
) -> Float[Array, "turbines"]:
    """Zero the raw action of turbines over the duty-cycle limit (``fidelity="floris"``)."""
    # Zeroing happens before the discrete mapping, so a zeroed discrete action
    # later maps to -yaw_step (a down move), not a hold, matching the FLORIS
    # reference's behavior. apply_action's "corrected" fidelity avoids this.
    return jnp.where(duty_over_active(accumulator, step_count), 0.0, action)


def command_from_action(
    action: Float[Array, "turbines"],
    *,
    yaw_step: float,
    control_mode: ControlMode,
) -> Float[Array, "turbines"]:
    """Map a raw action to a Δyaw command (deg).

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
    """Full single-farm yaw pipeline: duty limiter, action-to-command mapping, integration.

    ``fidelity="corrected"`` zeros the *mapped* command for duty-limited
    turbines (a hold), instead of ``"floris"``'s zero-the-raw-action behavior
    (which maps a zeroed discrete action to ``-yaw_step``, a down move);
    continuous control is unaffected either way.
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
    # Accumulator grows by the post-limiter, post-clip command magnitude (not
    # the ±40-saturated yaw change), matching WFCRL's reference convention.
    return AppliedAction(yaw=yaw_abs, accumulator=accumulator + jnp.abs(command))
