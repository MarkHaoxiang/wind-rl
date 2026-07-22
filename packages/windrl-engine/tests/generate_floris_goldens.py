"""Generate per-turbine GCH goldens from the latest FLORIS release for oracle retargeting.

Run isolated so the pinned FLORIS 3.5 in the project venv is never touched:

    uv run --isolated --no-project --with "floris==4.6.6" python \
        packages/windrl-engine/tests/generate_floris_goldens.py

FLORIS v4's `"defaults"` configuration is byte-identical, in every wake parameter
this project uses (sosfs / gauss velocity / gauss deflection / crespo_hernandez
constant 0.5, secondary-steering + yaw-added-recovery + transverse velocities on,
air density 1.225, TI 0.06, shear 0.12, veer 0), to the WFCRL GCH template that
drives the FLORIS 3.5 differential oracle. Only the built-in `nrel_5MW` turbine
library differs between the two releases, so any divergence these goldens expose
is a turbine-model change, not a wake-physics change.
"""

from pathlib import Path

import numpy as np
from floris import FlorisModel

Case = tuple[str, list[float], list[float], float, float, list[float]]

# turb3_row1 / ablaincourt layouts copied from windrl_engine.farm.layout (unavailable
# in the isolated env). 270 deg = wind from the west.
CASES: list[Case] = [
    (
        "row3_270_8_flat",
        [0.0, 504.0, 1008.0],
        [0.0, 0.0, 0.0],
        270.0,
        8.0,
        [0.0, 0.0, 0.0],
    ),
    (
        "row3_270_8_yaw",
        [0.0, 504.0, 1008.0],
        [0.0, 0.0, 0.0],
        270.0,
        8.0,
        [20.0, -15.0, 10.0],
    ),
    (
        "ablaincourt_240_10_flat",
        [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0],
        [274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0],
        240.0,
        10.0,
        [0.0] * 7,
    ),
]

AMBIENT_TI = 0.06


def solve(case: Case) -> dict[str, np.ndarray]:
    _, x, y, direction, speed, yaw = case
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=x,
        layout_y=y,
        wind_directions=[direction],
        wind_speeds=[speed],
        turbulence_intensities=[AMBIENT_TI],
        yaw_angles=np.asarray([yaw]),
    )
    fmodel.run()
    ff = fmodel.core.flow_field
    return {
        "powers": fmodel.get_turbine_powers().flatten(),
        "rotor_velocities": fmodel.turbine_average_velocities.flatten(),
        "turbulence_intensities": fmodel.get_turbine_TIs().flatten(),
        "u": ff.u[0],
        "v": ff.v[0],
        "w": ff.w[0],
    }


def main() -> None:
    import floris

    out: dict[str, np.ndarray] = {"floris_version": np.asarray(floris.__version__)}
    for case in CASES:
        for key, value in solve(case).items():
            out[f"{case[0]}/{key}"] = np.asarray(value, dtype=np.float64)

    goldens_dir = Path(__file__).parent / "goldens"
    goldens_dir.mkdir(exist_ok=True)
    path = goldens_dir / f"floris_v{floris.__version__}.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
