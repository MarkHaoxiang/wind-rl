"""The Designer contract: a pure `(key, batch) -> (batch, turbines, 2)` callable."""

from collections.abc import Callable

from jaxtyping import Array, Float, Key

#: A Designer maps a PRNG key and a batch size to a stack of turbine layouts,
#: xy coordinates in world meters. Designers are pure and deterministic per key;
#: batching lives inside the callable (vmap at the edge over a single-farm core).
#: A `Protocol` is deliberately not introduced: the current baselines are
#: closures with no observable state. Reintroduce one when a stateful designer
#: (learning/search: it must carry an updatable critic) actually exists.
Designer = Callable[[Key[Array, ""], int], Float[Array, "batch turbines 2"]]
