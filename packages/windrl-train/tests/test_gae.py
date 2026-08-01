import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from windrl_train.algo.ppo.types import Transition

gae_module = pytest.importorskip("windrl_train.algo.ppo.gae")

GAMMA = 0.9
LAMBDA = 0.8


def _col(values: list[float]) -> Float[Array, "steps 1 1"]:
    return jnp.asarray(values, dtype=jnp.float32).reshape(len(values), 1, 1)


def _transition(
    value: list[float],
    next_value: list[float],
    reward: list[float],
    done: list[bool],
) -> Transition:
    steps = len(value)
    return Transition(
        obs=jnp.zeros((steps, 1, 1, 1)),
        pre_tanh_action=jnp.zeros((steps, 1, 1)),
        action=jnp.zeros((steps, 1, 1)),
        log_prob=jnp.zeros((steps, 1, 1)),
        value=_col(value),
        next_value=_col(next_value),
        reward=_col(reward),
        done=jnp.asarray(done, dtype=bool).reshape(steps, 1, 1),
    )


CASE_A = _transition(
    value=[1.0, 2.0, 3.0],
    next_value=[2.0, 3.0, 4.0],
    reward=[1.0, 1.0, 1.0],
    done=[False, False, False],
)
CASE_A_ADVANTAGES = [3.85344, 2.852, 1.6]
CASE_A_TARGETS = [4.85344, 4.852, 4.6]

CASE_B = _transition(
    value=[1.0, 2.0, 3.0],
    next_value=[2.0, 5.0, 4.0],
    reward=[1.0, 1.0, 1.0],
    done=[False, True, False],
)
CASE_B_ADVANTAGES = [4.32, 3.5, 1.6]
CASE_B_TARGETS = [5.32, 5.5, 4.6]


def test_gae_matches_hand_computed_reference() -> None:
    advantages, targets = gae_module.gae(CASE_A, gamma=GAMMA, gae_lambda=LAMBDA)

    assert jnp.allclose(advantages, _col(CASE_A_ADVANTAGES), rtol=1e-5)
    assert jnp.allclose(targets, _col(CASE_A_TARGETS), rtol=1e-5)


def test_gae_truncation_bootstraps_next_value_and_fences_the_chain() -> None:
    advantages, targets = gae_module.gae(CASE_B, gamma=GAMMA, gae_lambda=LAMBDA)

    assert jnp.allclose(advantages, _col(CASE_B_ADVANTAGES), rtol=1e-5)
    assert jnp.allclose(targets, _col(CASE_B_TARGETS), rtol=1e-5)


def _lane(values: list[float]) -> Float[Array, "steps 2"]:
    return jnp.broadcast_to(jnp.asarray(values, dtype=jnp.float32)[:, None], (3, 2))


def test_gae_is_batched_over_envs_and_agents() -> None:
    value = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 3.0])[:, None, None], (3, 2, 2))
    reward = jnp.broadcast_to(jnp.asarray([1.0, 1.0, 1.0])[:, None, None], (3, 2, 2))
    next_value = jnp.stack([_lane([2.0, 3.0, 4.0]), _lane([2.0, 5.0, 4.0])], axis=1)
    done = jnp.stack(
        [
            _lane([0.0, 0.0, 0.0]).astype(bool),
            _lane([0.0, 1.0, 0.0]).astype(bool),
        ],
        axis=1,
    )
    traj = Transition(
        obs=jnp.zeros((3, 2, 2, 1)),
        pre_tanh_action=jnp.zeros((3, 2, 2)),
        action=jnp.zeros((3, 2, 2)),
        log_prob=jnp.zeros((3, 2, 2)),
        value=value,
        next_value=next_value,
        reward=reward,
        done=done,
    )

    advantages, targets = gae_module.gae(traj, gamma=GAMMA, gae_lambda=LAMBDA)

    assert jnp.allclose(advantages[:, 0, :], _lane(CASE_A_ADVANTAGES), rtol=1e-5)
    assert jnp.allclose(advantages[:, 1, :], _lane(CASE_B_ADVANTAGES), rtol=1e-5)
    assert jnp.allclose(targets[:, 0, :], _lane(CASE_A_TARGETS), rtol=1e-5)
    assert jnp.allclose(targets[:, 1, :], _lane(CASE_B_TARGETS), rtol=1e-5)


def test_gae_lambda_zero_is_one_step_td() -> None:
    advantages, _ = gae_module.gae(CASE_A, gamma=GAMMA, gae_lambda=0.0)

    deltas = [1.8, 1.7, 1.6]
    assert jnp.allclose(advantages, _col(deltas), rtol=1e-5)


@eqx.filter_jit
def _gae_jit(
    traj: Transition, gamma: float, gae_lambda: float
) -> tuple[Float[Array, "steps envs agents"], Float[Array, "steps envs agents"]]:
    return gae_module.gae(traj, gamma, gae_lambda)  # type: ignore[no-any-return]


def test_gae_jits() -> None:
    jitted_advantages, jitted_targets = _gae_jit(CASE_A, GAMMA, LAMBDA)
    plain_advantages, plain_targets = gae_module.gae(
        CASE_A, gamma=GAMMA, gae_lambda=LAMBDA
    )

    assert jnp.allclose(jitted_advantages, plain_advantages)
    assert jnp.allclose(jitted_targets, plain_targets)
