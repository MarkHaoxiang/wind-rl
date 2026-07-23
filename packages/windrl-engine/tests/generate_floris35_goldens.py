"""Freeze FLORIS 3.5 solver goldens for the differential reference tests.

The pinned differential oracle is FLORIS 3.5 driven through WFCRL's shipped GCH
template (secondary steering / yaw-added recovery / transverse velocities on,
crespo_hernandez constant 0.5). Once WFCRL leaves the dependency graph the live
oracle disappears, so these goldens capture its output once and the reference
tests assert against the frozen ``.npz`` forever after.

This generator embeds WFCRL's ``simulators/floris/inputs/template/case.yaml``
verbatim (only ``farm.layout_x/y`` and ``flow_field`` wind are overridden per
case, exactly as WFCRL's ``create_floris_case`` does) so it reproduces the same
numbers WFCRL did, with nothing but FLORIS 3.5 installed. Run isolated so the
project venv is never touched:

    uv run --isolated --no-project --with "floris==3.5" python \
        packages/windrl-engine/tests/generate_floris35_goldens.py

Layout coordinates are copied from ``windrl_engine.farm.layout`` (unavailable in
the isolated env); the reference tests re-assert them against the live layouts,
so any drift fails loudly.
"""

import copy
import tempfile
from pathlib import Path

import numpy as np
import yaml
from floris.tools import FlorisInterface

# WFCRL's simulators/floris/inputs/template/case.yaml, verbatim.
TEMPLATE: dict = {
    "name": "GCH",
    "description": "Base template for turbines using Gauss Curl Hybrid model",
    "floris_version": "v3.0.0",
    "logging": {
        "console": {"enable": True, "level": "WARNING"},
        "file": {"enable": False, "level": "WARNING"},
    },
    "solver": {"type": "turbine_grid", "turbine_grid_points": 3},
    "farm": {
        "layout_x": [0.0, 630.0, 1260.0],
        "layout_y": [0.0, 0.0, 0.0],
        "turbine_type": ["nrel_5MW"],
    },
    "flow_field": {
        "air_density": 1.225,
        "reference_wind_height": -1,
        "turbulence_intensity": 0.06,
        "wind_directions": [270.0],
        "wind_shear": 0.12,
        "wind_speeds": [8.0],
        "wind_veer": 0.0,
    },
    "wake": {
        "model_strings": {
            "combination_model": "sosfs",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "velocity_model": "gauss",
        },
        "enable_secondary_steering": True,
        "enable_yaw_added_recovery": True,
        "enable_transverse_velocities": True,
        "wake_deflection_parameters": {
            "gauss": {
                "ad": 0.0,
                "alpha": 0.58,
                "bd": 0.0,
                "beta": 0.077,
                "dm": 1.0,
                "ka": 0.38,
                "kb": 0.004,
            },
            "jimenez": {"ad": 0.0, "bd": 0.0, "kd": 0.05},
        },
        "wake_velocity_parameters": {
            "cc": {
                "a_s": 0.179367259,
                "b_s": 0.0118889215,
                "c_s1": 0.0563691592,
                "c_s2": 0.13290157,
                "a_f": 3.11,
                "b_f": -0.68,
                "c_f": 2.41,
                "alpha_mod": 1.0,
            },
            "gauss": {"alpha": 0.58, "beta": 0.077, "ka": 0.38, "kb": 0.004},
            "jensen": {"we": 0.05},
        },
        "wake_turbulence_parameters": {
            "crespo_hernandez": {
                "initial": 0.1,
                "constant": 0.5,
                "ai": 0.8,
                "downstream": -0.32,
            }
        },
    },
}

# Layout coordinates copied from windrl_engine.farm.layout.
TURB3_ROW1_X = [0.0, 504.0, 1008.0]
TURB3_ROW1_Y = [0.0, 0.0, 0.0]
ABLAINCOURT_X = [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0]
ABLAINCOURT_Y = [274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0]
HORNS_REV2_X = [
    3586.690071,
    3038.283676,
    2502.705501,
    1957.5061600000001,
    1421.927985,
    873.52159,
    341.150469,
    3548.2054120000003,
    3003.0060719999997,
    2457.8067309999997,
    1922.228556,
    1380.236271,
    831.829876,
    286.630535,
    3551.412467,
    3022.2484010000003,
    2473.842006,
    1928.642666,
    1383.443326,
    838.2439860000001,
    293.044645,
    3609.139456,
    3063.9401159999998,
    2531.568995,
    1986.369655,
    1444.37737,
    908.7991939999999,
    363.599854,
    3708.5581589999997,
    3163.358819,
    2627.780643,
    2101.823633,
    1563.038403,
    1017.839062,
    488.674997,
    3849.668576,
    3310.8833459999996,
    2778.5122260000003,
    2249.34816,
    1720.184095,
    1191.020029,
    671.477129,
    4019.6424879999995,
    3496.892533,
    2977.349632,
    2454.599676,
    1938.263831,
    1428.3420950000002,
    905.592139,
    4240.929279,
    3731.007544,
    3237.1210819999997,
    2727.199347,
    2214.0705559999997,
    1704.14882,
    1197.434139,
    4513.5289490000005,
    4013.228378,
    3516.134862,
    3028.662511,
    2534.77605,
    2037.682534,
    1550.210183,
    4824.613279,
    4340.347983,
    3856.082686,
    3378.2315,
    2903.587368,
    2432.1502920000003,
    1954.299105,
    5170.975213000001,
    4702.745191,
    4253.757498999999,
    3788.734532,
    3333.332731,
    2868.309764,
    2403.2867969999998,
    5562.235916,
    5119.662334,
    4673.881697,
    4231.308115,
    3798.355697,
    3355.782115,
    2910.0014779999997,
    5988.774223,
    5565.442971,
    5138.904663,
    4725.194576,
    4305.070378,
    3881.739126,
    3455.200818,
]
HORNS_REV2_Y = [
    1990.3730449999998,
    1945.447252,
    1897.3124750000002,
    1839.550742,
    1794.62495,
    1749.6991580000001,
    1701.56438,
    2667.468913,
    2657.841957,
    2648.2150020000004,
    2622.543121,
    2616.12515,
    2612.916165,
    2600.080225,
    3376.6546329999996,
    3383.072603,
    3411.95347,
    3440.834336,
    3456.8792620000004,
    3488.969114,
    3492.178099,
    4056.959487,
    4105.094264,
    4172.482952,
    4239.871641,
    4288.006418,
    4355.395106,
    4403.5298840000005,
    4740.473325,
    4846.369836,
    4933.012435000001,
    5016.446049,
    5125.551544,
    5212.194144,
    5311.672684,
    5427.196149,
    5552.34657,
    5674.288006000001,
    5809.065383,
    5934.2158039999995,
    6056.1572400000005,
    6178.0986760000005,
    6088.247092,
    6245.487365,
    6412.354593,
    6582.430805999999,
    6742.880064,
    6906.538307,
    7063.77858,
    6755.716005,
    6945.046128999999,
    7140.794224,
    7317.288407999999,
    7519.454473,
    7708.784597,
    7907.741677,
    7391.095066,
    7609.306057000001,
    7837.144002999999,
    8074.608905,
    8286.401925,
    8520.657842,
    8751.704773,
    7994.384276,
    8270.357,
    8520.657842,
    8787.00361,
    9034.095467000001,
    9274.769354,
    9550.742078000001,
    8597.673486,
    8889.691135,
    9168.872844,
    9451.263538,
    9743.281186999999,
    10028.88087,
    10314.48055,
    9181.708784999999,
    9483.353389,
    9804.251905,
    10131.56839,
    10436.421980000001,
    10763.73847,
    11084.636980000001,
    9720.818291,
    10064.1797,
    10407.54112,
    10750.90253,
    11094.263939999999,
    11453.67028,
    11800.240670000001,
]

# (case id, layout_x, layout_y, direction, speed, yaw)
CASES = [
    ("3t-270-8-flat", TURB3_ROW1_X, TURB3_ROW1_Y, 270.0, 8.0, [0.0, 0.0, 0.0]),
    ("3t-270-8-yaw", TURB3_ROW1_X, TURB3_ROW1_Y, 270.0, 8.0, [20.0, -15.0, 10.0]),
    ("3t-240-11-flat", TURB3_ROW1_X, TURB3_ROW1_Y, 240.0, 11.0, [0.0, 0.0, 0.0]),
    ("3t-83.5-8-flat", TURB3_ROW1_X, TURB3_ROW1_Y, 83.5, 8.0, [0.0, 0.0, 0.0]),
    ("7t-270-8-flat", ABLAINCOURT_X, ABLAINCOURT_Y, 270.0, 8.0, [0.0] * 7),
    (
        "7t-270-11-yaw",
        ABLAINCOURT_X,
        ABLAINCOURT_Y,
        270.0,
        11.0,
        [20.0, -15.0, 10.0, 0.0, 0.0, 0.0, 0.0],
    ),
    ("91t-270-8-flat", HORNS_REV2_X, HORNS_REV2_Y, 270.0, 8.0, [0.0] * 91),
]


def solve(x, y, direction, speed, yaw):
    config = copy.deepcopy(TEMPLATE)
    config["farm"]["layout_x"] = list(x)
    config["farm"]["layout_y"] = list(y)
    config["flow_field"]["wind_directions"] = [direction]
    config["flow_field"]["wind_speeds"] = [speed]

    # FlorisInterface(configuration=dict) is accepted, but writing the yaml the
    # same way WFCRL's create_floris_case does removes any doubt about dict-vs-
    # file schema handling in FLORIS 3.5.
    out = Path(tempfile.mkdtemp(prefix="floris35_golden_")) / "case.yaml"
    with open(out, "w") as fp:
        yaml.safe_dump(config, fp)

    fi = FlorisInterface(str(out))
    fi.reinitialize(wind_speeds=[speed], wind_directions=[direction])
    fi.calculate_wake(yaw_angles=np.asarray(yaw)[None, None, :])
    ff = fi.floris.flow_field
    return {
        "layout_x": np.asarray(fi.layout_x, dtype=np.float64),
        "layout_y": np.asarray(fi.layout_y, dtype=np.float64),
        "u": np.asarray(ff.u[0, 0], dtype=np.float64),
        "v": np.asarray(ff.v[0, 0], dtype=np.float64),
        "w": np.asarray(ff.w[0, 0], dtype=np.float64),
        "ti": np.asarray(
            ff.turbulence_intensity_field.squeeze(axis=(0, 1)), dtype=np.float64
        ).reshape(-1),
        "powers": np.asarray(fi.get_turbine_powers().flatten(), dtype=np.float64),
        "yaw": np.asarray(yaw, dtype=np.float64),
    }


def main() -> None:
    import floris

    out: dict[str, np.ndarray] = {"floris_version": np.asarray(floris.__version__)}
    for case_id, x, y, direction, speed, yaw in CASES:
        for key, value in solve(x, y, direction, speed, yaw).items():
            out[f"{case_id}/{key}"] = value

    goldens_dir = Path(__file__).parent / "goldens"
    goldens_dir.mkdir(exist_ok=True)
    path = goldens_dir / "floris_v3.5.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
