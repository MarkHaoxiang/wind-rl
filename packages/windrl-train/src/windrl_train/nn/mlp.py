import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    """Feedforward net over the trailing axis: any leading batch shape works with no vmap."""

    weights: list[Float[Array, "fan_in fan_out"]]
    biases: list[Float[Array, "fan_out"]]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        sizes = [in_size, *([width] * depth), out_size]
        keys = jax.random.split(key, len(sizes) - 1)
        self.weights = [
            jax.random.uniform(layer_key, (fan_in, fan_out), minval=-1.0, maxval=1.0)
            / jnp.sqrt(fan_in)
            for layer_key, fan_in, fan_out in zip(
                keys, sizes[:-1], sizes[1:], strict=True
            )
        ]
        self.biases = [jnp.zeros(fan_out) for fan_out in sizes[1:]]

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        num_layers = len(self.weights)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ weight + bias
            if i < num_layers - 1:
                x = jax.nn.silu(x)
        return x
