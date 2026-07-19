"""Wind farm scenario configuration and the real-farm layout registry.

The real-farm registry wraps ``wfcrl.environments.data_cases.named_cases_dictionary``
rather than duplicating its coordinate tables: wfcrl already ships the fixed
turbine layouts (Horns Rev 1/2, Ormonde, WMR, Ablaincourt, TCRWP, and a few
synthetic row layouts) as ``FastFarmCase``/``FlorisCase`` dataclass instances
keyed by name (with a trailing underscore, e.g. ``"HornsRev1_"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from wind_rl.config import Config


class ScenarioConfig(Config):
    name: str
    n_turbines: int = Field(ge=1)
    max_steps: int = Field(gt=0)
    map_x_length: float = Field(gt=0)
    map_y_length: float = Field(gt=0)
    min_distance_between_turbines: float = Field(gt=0)


def _named_cases() -> dict[str, list[Any]]:
    # Imported lazily so importing wind_rl.scenario never requires wfcrl unless
    # the real-farm registry is actually used.
    from wfcrl.environments.data_cases import named_cases_dictionary

    return named_cases_dictionary  # type: ignore[no-any-return]


def list_real_farms() -> list[str]:
    """List the names of real (and reference) farm layouts known to wfcrl."""
    return sorted(key.rstrip("_") for key in _named_cases())


def real_farm_layout(name: str) -> NDArray[np.float64]:
    """Return the ``(N, 2)`` xy turbine coordinates for a named wfcrl farm case.

    ``name`` may be given with or without wfcrl's trailing underscore
    (e.g. both ``"HornsRev1"`` and ``"HornsRev1_"`` resolve to the same case).
    """
    cases = _named_cases()
    key = name if name in cases else f"{name}_"
    if key not in cases:
        raise KeyError(
            f"Unknown real farm layout {name!r}. Available: {list_real_farms()}"
        )

    # Each entry is [FastFarmCase, FlorisCase]; both share the same coordinates,
    # so the FlorisCase is used arbitrarily.
    farm_case = cases[key][1]
    xcoords, ycoords = farm_case.xcoords, farm_case.ycoords
    if callable(xcoords) or callable(ycoords):
        raise TypeError(f"Real farm {name!r} uses a procedural (not fixed) layout")
    return np.column_stack(
        [np.asarray(xcoords, dtype=np.float64), np.asarray(ycoords, dtype=np.float64)]
    )
