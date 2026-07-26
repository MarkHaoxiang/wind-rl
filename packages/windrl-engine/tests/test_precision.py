"""Precision contract of the wake solve, exercised in subprocesses.

The session-wide ``conftest.py`` enables ``jax_enable_x64`` before any engine
module is imported, so the two configurations that matter here -- x64 switched
on *after* ``farm.turbine`` is imported, and x64 never switched on at all --
are only reachable from a fresh interpreter.
"""

import json
import subprocess
import sys

import numpy as np

# FLORIS 4.6.6 turb3_row1 @ 270 deg / 8 m/s / zero yaw; same anchor as
# test_reference_solver.py's regression guard.
ANCHOR_POWERS = np.array([1753954.45917917, 356384.89485703, 344414.69675443])

_SOLVE_SCRIPT = """
import json, sys
import numpy as np
import windrl_engine.farm.turbine  # noqa: F401 -- imported before the x64 switch
import jax

if {enable_x64}:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from windrl_engine.farm.layout import turb3_row1
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm

yaw = jnp.zeros(3)
wind = WindCondition(speed=jnp.asarray(8.0), direction=jnp.asarray(270.0))
solution = solve_farm(turb3_row1(), wind, yaw)
powers = np.asarray(turbine_powers(solution.u, yaw), dtype=np.float64)
json.dump({{"powers": powers.tolist(), "dtype": str(solution.u.dtype)}}, sys.stdout)
"""


def _solve_in_subprocess(*, enable_x64: bool) -> tuple[np.ndarray, str]:
    completed = subprocess.run(
        [sys.executable, "-c", _SOLVE_SCRIPT.format(enable_x64=enable_x64)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    return np.asarray(result["powers"]), result["dtype"]


def test_x64_enabled_after_turbine_import_still_matches_the_floris_anchor() -> None:
    powers, dtype = _solve_in_subprocess(enable_x64=True)
    assert dtype == "float64"
    np.testing.assert_allclose(powers, ANCHOR_POWERS, rtol=1e-9)


def test_float32_solve_stays_within_1e_6_of_the_floris_anchor() -> None:
    powers, dtype = _solve_in_subprocess(enable_x64=False)
    assert dtype == "float32"
    np.testing.assert_allclose(powers, ANCHOR_POWERS, rtol=1e-6)
