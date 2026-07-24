"""farm/ is fully implemented (design doc §"Package tree"); these run now."""

import math
from importlib.resources import files
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from windrl_engine.farm.layout import horns_rev2, row_layout
from windrl_engine.farm.turbine import D, ct_lookup, nrel5mw_v4, power_lookup
from windrl_engine.farm.wind import sample_wind

N_WIND_SAMPLES = 20_000


def _wind_samples() -> tuple[jax.Array, jax.Array]:
    # `Key[Array, ""]` (design doc's fixed signature) is a scalar typed key,
    # i.e. `jax.random.key(...)`, not the legacy `uint32[2]` PRNGKey array.
    keys = jax.random.split(jax.random.key(0), N_WIND_SAMPLES)
    speed, direction = jax.vmap(sample_wind)(keys)
    return speed, direction


def test_sample_wind_respects_clip_bounds() -> None:
    speed, direction = _wind_samples()
    assert bool(jnp.all(speed >= 0.0))
    assert bool(jnp.all(speed <= 28.0))
    assert bool(jnp.all(direction >= 0.0))
    assert bool(jnp.all(direction <= 360.0))


def test_sample_wind_speed_matches_8_weibull_8_moments() -> None:
    speed, _ = _wind_samples()
    # Weibull(k=8, scale=1) moments via the gamma function; sample is 8·Weibull(8).
    k = 8.0
    weibull_mean = math.gamma(1.0 + 1.0 / k)
    weibull_var = math.gamma(1.0 + 2.0 / k) - weibull_mean**2
    expected_mean = 8.0 * weibull_mean
    expected_std = 8.0 * math.sqrt(weibull_var)
    assert expected_mean == pytest.approx(7.535, abs=2e-3)

    sample_mean = float(jnp.mean(speed))
    sample_std = float(jnp.std(speed))
    # Loose statistical tolerance: standard error of the mean at n=20000 is
    # expected_std/sqrt(n) ~= 0.01; allow a generous 20x margin.
    se = expected_std / math.sqrt(N_WIND_SAMPLES)
    assert sample_mean == pytest.approx(expected_mean, abs=20 * se)
    assert sample_std == pytest.approx(expected_std, rel=0.05)


def test_sample_wind_direction_matches_normal_270_20_moments() -> None:
    _, direction = _wind_samples()
    sample_mean = float(jnp.mean(direction))
    sample_std = float(jnp.std(direction))
    se = 20.0 / math.sqrt(N_WIND_SAMPLES)
    assert sample_mean == pytest.approx(270.0, abs=20 * se)
    assert sample_std == pytest.approx(20.0, rel=0.05)


def test_row_layout_spacing_is_4_rotor_diameters() -> None:
    layout = row_layout(5)
    diffs = jnp.diff(layout.x)
    assert jnp.allclose(diffs, 4 * D).item()
    assert jnp.allclose(layout.y, 0.0).item()


def test_horns_rev2_has_91_turbines() -> None:
    layout = horns_rev2()
    assert layout.x.shape == (91,)
    assert layout.y.shape == (91,)


# --- nrel_5MW v4 tables: floris yaml <-> turbine.py consistency --------------


def _floris_nrel5mw_yaml() -> dict[str, Any]:
    text = (files("floris") / "turbine_library" / "nrel_5MW.yaml").read_text()
    return yaml.safe_load(text)


def test_turbine_spec_matches_floris_yaml() -> None:
    # turbine.py reads floris's packaged yaml at import; assert against that same
    # source so upstream drift is caught without a committed intermediate.
    spec = nrel5mw_v4()
    yml = _floris_nrel5mw_yaml()
    table = yml["power_thrust_table"]

    assert spec.wind_speed_table.shape == (54,)
    np.testing.assert_array_equal(
        np.asarray(spec.wind_speed_table), np.asarray(table["wind_speed"])
    )
    np.testing.assert_array_equal(
        np.asarray(spec.thrust_table), np.asarray(table["thrust_coefficient"])
    )
    np.testing.assert_array_equal(
        np.asarray(spec.power_table), np.asarray(table["power"])
    )

    assert spec.rotor_diameter == yml["rotor_diameter"]
    assert spec.hub_height == yml["hub_height"]
    assert spec.pP == table["cosine_loss_exponent_yaw"]
    assert spec.tsr == yml["TSR"]
    assert spec.ref_density == table["ref_air_density"]
    assert (
        spec.generator_efficiency
        == table["controller_dependent_turbine_parameters"]["generator_efficiency"]
    )


def test_ct_lookup_matches_table_at_a_nonzero_node() -> None:
    spec = nrel5mw_v4()
    ws = spec.wind_speed_table[16]  # 8.0 m/s, C_t inside the (1e-4, 0.9999) clip
    assert float(ct_lookup(spec, ws)) == pytest.approx(
        float(spec.thrust_table[16]), abs=1e-9
    )


def test_ct_lookup_clips_out_of_range_to_fill() -> None:
    spec = nrel5mw_v4()
    assert float(ct_lookup(spec, jnp.asarray(-10.0))) == pytest.approx(
        0.0001, abs=1e-12
    )
    assert float(ct_lookup(spec, jnp.asarray(100.0))) == pytest.approx(
        0.0001, abs=1e-12
    )


def test_power_lookup_scales_abs_kw_table_to_watts() -> None:
    spec = nrel5mw_v4()
    ws = spec.wind_speed_table[16]
    expected = float(spec.power_table[16]) * spec.power_scale
    assert float(power_lookup(spec, ws)) == pytest.approx(expected, rel=1e-9)


def test_power_lookup_fill_out_of_range_is_zero() -> None:
    spec = nrel5mw_v4()
    assert float(power_lookup(spec, jnp.asarray(-10.0))) == pytest.approx(0.0, abs=1e-9)
    assert float(power_lookup(spec, jnp.asarray(100.0))) == pytest.approx(0.0, abs=1e-9)
