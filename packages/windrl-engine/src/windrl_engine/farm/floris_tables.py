"""NREL-5MW turbine tables read straight from the installed FLORIS package.

Kept out of the jaxtyping/beartype import hook (see ``tests/conftest.py``): the
loader is annotated with ``npt.NDArray``, which the hook rejects.
"""

from importlib.resources import files

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]


def load_nrel5mw_v4() -> dict[str, npt.NDArray[np.float64]]:
    """FLORIS 4.6.6 ``nrel_5MW`` power/thrust tables and geometry (cosine-loss model)."""
    text = (files("floris") / "turbine_library" / "nrel_5MW.yaml").read_text()
    spec = yaml.safe_load(text)
    table = spec["power_thrust_table"]
    return {
        "wind_speed": np.asarray(table["wind_speed"], dtype=np.float64),
        "power_kw": np.asarray(table["power"], dtype=np.float64),
        "thrust": np.asarray(table["thrust_coefficient"], dtype=np.float64),
        "rotor_diameter": np.asarray(spec["rotor_diameter"], dtype=np.float64),
        "hub_height": np.asarray(spec["hub_height"], dtype=np.float64),
        "pP": np.asarray(table["cosine_loss_exponent_yaw"], dtype=np.float64),
        "tsr": np.asarray(spec["TSR"], dtype=np.float64),
        "ref_density": np.asarray(table["ref_air_density"], dtype=np.float64),
        "generator_efficiency": np.asarray(
            table["controller_dependent_turbine_parameters"]["generator_efficiency"],
            dtype=np.float64,
        ),
    }
