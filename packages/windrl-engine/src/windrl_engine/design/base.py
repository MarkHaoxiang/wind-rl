"""The Designer contract: a pure `(key, batch) -> (batch, turbines, 2)` callable."""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float, Key


@runtime_checkable
class Designer(Protocol):
    """A pure map from a PRNG key and batch size to stacked turbine layouts.

    Layouts are xy coordinates in world meters, deterministic per key; batching
    lives inside the callable (vmap at the edge over a single-farm core).
    """

    def __call__(
        self, key: Key[Array, ""], batch_size: int
    ) -> Float[Array, "batch turbines 2"]: ...
