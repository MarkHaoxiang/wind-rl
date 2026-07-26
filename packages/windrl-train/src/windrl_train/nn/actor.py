import distrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from windrl_train.nn.mlp import MLP

_INITIAL_STD = 0.5


class Actor(eqx.Module):
    """Per-agent scalar delta-yaw policy, applied independently over the agent axis."""

    torso: MLP
    log_std: Float[Array, ""]
    action_scale: float = eqx.field(static=True)

    def __init__(
        self,
        feat_size: int,
        width: int,
        depth: int,
        action_scale: float,
        *,
        key: Key[Array, ""],
    ) -> None:
        self.torso = MLP(feat_size, 1, width, depth, key=key)
        self.log_std = jnp.log(jnp.asarray(_INITIAL_STD))
        self.action_scale = action_scale

    def __call__(self, feats: Float[Array, "... feat"]) -> distrax.Distribution:
        mu = self.torso(feats)[..., 0]
        base = distrax.Normal(loc=mu, scale=jnp.exp(self.log_std))
        # Chain applies bijectors right-to-left: Tanh first (into (-1, 1)),
        # then the scale — so samples land strictly inside [-scale, scale].
        bijector = distrax.Chain(
            [distrax.ScalarAffine(shift=0.0, scale=self.action_scale), distrax.Tanh()]
        )
        return distrax.Transformed(distribution=base, bijector=bijector)

    def mode(self, feats: Float[Array, "... feat"]) -> Float[Array, "..."]:
        mu = self.torso(feats)[..., 0]
        return jnp.tanh(mu) * self.action_scale
