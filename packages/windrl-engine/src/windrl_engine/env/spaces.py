import dataclasses

from jaxtyping import Array, Float


@dataclasses.dataclass(frozen=True)
class Box:
    """A bounded float array space (shape excludes any leading batch axis).

    ``low``/``high`` are scalar (broadcast over the space) or per-element arrays.
    """

    shape: tuple[int, ...]
    low: float | Float[Array, " ..."]
    high: float | Float[Array, " ..."]
    dtype: str = "float64"


@dataclasses.dataclass(frozen=True)
class MultiDiscrete:
    """Per-element discrete space; ``nvec[i]`` values ``{0, ..., nvec[i] - 1}``."""

    nvec: tuple[int, ...]


Space = Box | MultiDiscrete
