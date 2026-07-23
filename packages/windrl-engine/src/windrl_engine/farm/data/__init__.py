"""Committed package data: turbine tables generated from FLORIS (see tests/generate_turbine_data.py)."""

from importlib.resources import files

import numpy as np
import numpy.typing as npt


def load_nrel5mw_v4() -> dict[str, npt.NDArray[np.float64]]:
    """FLORIS 4.6.6 nrel_5MW tables + scalars, frozen in ``nrel5mw_v4.npz``."""
    resource = files("windrl_engine.farm.data") / "nrel5mw_v4.npz"
    with resource.open("rb") as fh, np.load(fh) as npz:
        return {key: npz[key] for key in npz.files if key != "floris_version"}
