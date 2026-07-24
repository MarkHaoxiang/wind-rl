"""Freeze the NREL-5MW turbine tables from FLORIS 4.6.6's packaged YAML.

``windrl_engine.farm.turbine`` ships the FLORIS 4.6.6 ``nrel_5MW`` power/thrust
tables and geometry as committed package data (``farm/data/nrel5mw_v4.npz``)
rather than hand-transcribed literals, so the numbers cannot silently drift from
their upstream source. This generator extracts them from the ``floris`` wheel's
own ``turbine_library/nrel_5MW.yaml`` and writes the npz. Run isolated so the
project venv is never touched:

    uv run --isolated --no-project --with "floris==4.6.6" python \
        packages/windrl-engine/tests/generate_turbine_data.py

The 54-point tables are the "cosine-loss" operation model: ``power`` is absolute
electrical output in kW, ``thrust_coefficient`` is C_t. ``test_farm.py`` asserts
the shipped npz matches what ``turbine.py`` exposes, so a regenerated artifact
that disagrees fails loudly.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]


def _find_scalar(node: object, key: str) -> float:
    """Depth-first search for ``key`` anywhere in the parsed YAML tree."""
    if isinstance(node, dict):
        if key in node and not isinstance(node[key], (dict, list)):
            return float(node[key])
        for value in node.values():
            found = _find_scalar(value, key)
            if found is not None:
                return found
    return None  # type: ignore[return-value]


def main() -> None:
    import floris

    text = (files("floris") / "turbine_library" / "nrel_5MW.yaml").read_text()
    spec = yaml.safe_load(text)
    table = spec["power_thrust_table"]

    out = {
        "floris_version": np.asarray(floris.__version__),
        "wind_speed": np.asarray(table["wind_speed"], dtype=np.float64),
        "power_kw": np.asarray(table["power"], dtype=np.float64),
        "thrust": np.asarray(table["thrust_coefficient"], dtype=np.float64),
        "rotor_diameter": np.asarray(spec["rotor_diameter"], dtype=np.float64),
        "hub_height": np.asarray(spec["hub_height"], dtype=np.float64),
        "pP": np.asarray(
            _find_scalar(spec, "cosine_loss_exponent_yaw"), dtype=np.float64
        ),
        "tsr": np.asarray(_find_scalar(spec, "TSR"), dtype=np.float64),
        "ref_density": np.asarray(
            _find_scalar(table, "ref_air_density"), dtype=np.float64
        ),
        "generator_efficiency": np.asarray(
            _find_scalar(spec, "generator_efficiency"), dtype=np.float64
        ),
    }

    n = out["wind_speed"].shape[0]
    assert out["power_kw"].shape == (n,) and out["thrust"].shape == (n,), (
        "power/thrust tables must match wind_speed length"
    )

    data_dir = Path(__file__).parents[1] / "src" / "windrl_engine" / "farm" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "nrel5mw_v4.npz"
    np.savez_compressed(path, **out)

    print(f"floris {floris.__version__}, {n}-point tables")
    for key in (
        "rotor_diameter",
        "hub_height",
        "pP",
        "tsr",
        "ref_density",
        "generator_efficiency",
    ):
        print(f"  {key} = {out[key]}")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
