import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from windrl_train.nn.mlp import MLP


class Critic(eqx.Module):
    """Per-agent state-value estimate, applied independently over the agent axis."""

    torso: MLP

    def __init__(
        self, feat_size: int, width: int, depth: int, *, key: PRNGKeyArray
    ) -> None:
        self.torso = MLP(feat_size, 1, width, depth, key=key)

    def __call__(self, feats: Float[Array, "... feat"]) -> Float[Array, "..."]:
        return self.torso(feats)[..., 0]
