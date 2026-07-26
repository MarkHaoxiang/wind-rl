"""On-demand hub-height wake fields for a recorded episode, cached per frame."""

import functools
from collections.abc import Callable
from typing import Final, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float

from windrl_engine.farm.layout import FarmLayout
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.solver import Fidelity
from windrl_engine.viz.plane import Extent, horizontal_slice, padded_extent
from windrl_engine.viz.record import EpisodeRecord

DEFAULT_RESOLUTION: Final = (240, 240)
# ~230 kB/frame at DEFAULT_RESOLUTION: enough for a scrub window, not a whole episode.
CACHED_FRAMES: Final = 64


@functools.partial(
    jax.jit, static_argnames=("resolution", "height", "bounds", "fidelity")
)
def _hub_field(
    layout: FarmLayout,
    wind: WindCondition,
    yaw: Float[Array, "turbines"],
    *,
    resolution: tuple[int, int],
    height: float,
    bounds: Extent,
    fidelity: Fidelity,
) -> Float[Array, "ny nx"]:
    field, _ = horizontal_slice(
        layout,
        wind,
        yaw,
        height=height,
        bounds=bounds,
        resolution=resolution,
        fidelity=fidelity,
    )
    return field


class EpisodeFields:
    """Lazily solves the z=hub-height u-field for each recorded frame, most recent cached.

    The domain extent is layout-only, so it is fixed across the episode; per
    frame only the wind and yaw change. ``field_at`` returns row-major float32
    (ny, nx) with ``origin='lower'`` (row 0 is ``ymin``).
    """

    def __init__(
        self,
        record: EpisodeRecord,
        *,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        height: float | None = None,
    ) -> None:
        self._layout = FarmLayout(
            x=jnp.asarray(record.layout_x), y=jnp.asarray(record.layout_y)
        )
        self._yaw = jnp.asarray(record.yaw)
        self._speed = jnp.asarray(record.wind_speed)
        self._direction = jnp.asarray(record.wind_direction)
        self._resolution = resolution
        self._height = record.hub_height if height is None else height
        self._fidelity = record.fidelity
        self.extent: Extent = padded_extent(self._layout, record.rotor_diameter)
        self._cached_field: Callable[[int], npt.NDArray[np.float32]] = (
            functools.lru_cache(maxsize=CACHED_FRAMES)(self._render)
        )

    @property
    def n_frames(self) -> int:
        return int(self._yaw.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny = self._resolution  # horizontal_slice(resolution=(nx, ny)) -> (ny, nx)
        return ny, nx

    def _solve(self, frame: int) -> Float[Array, "ny nx"]:
        wind = WindCondition(speed=self._speed[frame], direction=self._direction[frame])
        return cast(
            Float[Array, "ny nx"],
            _hub_field(
                self._layout,
                wind,
                self._yaw[frame],
                resolution=self._resolution,
                height=self._height,
                bounds=self.extent,
                fidelity=self._fidelity,
            ),
        )

    def _render(self, frame: int) -> npt.NDArray[np.float32]:
        return np.asarray(self._solve(frame), dtype=np.float32)

    def field_at(self, frame: int) -> npt.NDArray[np.float32]:
        return self._cached_field(frame)
