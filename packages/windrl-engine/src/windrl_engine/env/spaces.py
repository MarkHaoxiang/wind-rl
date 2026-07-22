import dataclasses


@dataclasses.dataclass(frozen=True)
class Box:
    """A bounded float array space (shape excludes any leading batch axis)."""

    shape: tuple[int, ...]
    low: float
    high: float
    dtype: str = "float64"


@dataclasses.dataclass(frozen=True)
class MultiDiscrete:
    """Per-element discrete space; ``nvec[i]`` values ``{0, ..., nvec[i] - 1}``."""

    nvec: tuple[int, ...]


Space = Box | MultiDiscrete
