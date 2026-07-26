"""farm/: the wind sampler's distribution, the site layouts, and the NREL-5MW tables."""

import math
from importlib.resources import files
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from windrl_engine.farm.layout import ROW_SPACING, horns_rev2, row_layout
from windrl_engine.farm.turbine import (
    CT_MAX,
    POWER_SCALE,
    ct_lookup,
    nrel5mw_v4,
    power_lookup,
)
from windrl_engine.farm.wind import sample_wind

N_WIND_SAMPLES = 20_000


def _wind_samples() -> tuple[jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.key(0), N_WIND_SAMPLES)
    speed, direction = jax.vmap(sample_wind)(keys)
    return speed, direction


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
    # The [0, 28] clip is an inf-guard for a uniform draw of exactly 0.0, not a
    # physical cap: 28 m/s is ~30 sigma out (P ~ 1e-9780), so a sample that
    # reaches either bound means the sampler, not the tail.
    assert bool(jnp.all((speed >= 0.0) & (speed <= 28.0)))


def test_sample_wind_direction_matches_normal_270_20_moments() -> None:
    _, direction = _wind_samples()
    sample_mean = float(jnp.mean(direction))
    sample_std = float(jnp.std(direction))
    se = 20.0 / math.sqrt(N_WIND_SAMPLES)
    assert sample_mean == pytest.approx(270.0, abs=20 * se)
    assert sample_std == pytest.approx(20.0, rel=0.05)
    assert bool(jnp.all((direction >= 0.0) & (direction <= 360.0)))


def test_row_layout_spacing_is_the_wfcrl_nominal_four_diameters() -> None:
    layout = row_layout(5)
    diffs = jnp.diff(layout.x)
    assert jnp.allclose(diffs, ROW_SPACING).item()
    assert jnp.allclose(layout.y, 0.0).item()


def test_horns_rev2_is_thirteen_fanned_rows_of_seven() -> None:
    layout = horns_rev2()
    assert layout.x.shape == (91,)
    x = np.asarray(layout.x).reshape(13, 7)
    y = np.asarray(layout.y).reshape(13, 7)

    in_row = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    assert np.all((in_row > 520.0) & (in_row < 560.0))

    # The rows fan out downwind, so row-to-row spacing grows monotonically from
    # the leading column to the trailing one (692 m -> 903 m on average).
    row_gap = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).mean(axis=0)
    assert np.all(np.diff(row_gap) > 0.0)
    assert row_gap[0] == pytest.approx(692.0, abs=1.0)
    assert row_gap[-1] == pytest.approx(903.0, abs=1.0)


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


def test_floris_yaml_still_makes_the_tilt_correction_a_no_op() -> None:
    # physics/thrust.py drops the tilt term as unity, which only holds while
    # nrel_5MW ships tilt correction off and its reference tilt matches the
    # (constant) floating tilt table; flipping either upstream would silently
    # break parity.
    yml = _floris_nrel5mw_yaml()
    assert yml["correct_cp_ct_for_tilt"] is False
    assert yml["power_thrust_table"]["ref_tilt"] == 5.0


def test_ct_lookup_interpolates_linearly_between_table_nodes() -> None:
    # Midway between two nodes, not on one: nearest-neighbour or step lookup
    # would land on 0.7871 or 0.7858 instead, both far outside 1e-9.
    spec = nrel5mw_v4()
    lower, upper = 16, 17  # 8.0 and 9.0 m/s, inside the C_t clip
    midpoint = jnp.asarray(
        0.5 * (spec.wind_speed_table[lower] + spec.wind_speed_table[upper])
    )
    expected = 0.5 * (spec.thrust_table[lower] + spec.thrust_table[upper])

    assert float(ct_lookup(spec, midpoint)) == pytest.approx(float(expected), abs=1e-9)


def test_ct_lookup_clips_the_one_table_node_that_exceeds_the_upper_bound() -> None:
    # nrel_5MW's C_t table is physically inconsistent at cut-in: it lists 1.132
    # at 3.0 m/s, above the Betz-admissible 1. That single node is the only
    # place the CT_MAX clip does anything, and dropping the clip would push a
    # C_t > 1 into the axial-induction sqrt.
    spec = nrel5mw_v4()
    assert float(spec.thrust_table[2]) > CT_MAX
    assert float(spec.wind_speed_table[2]) == 3.0
    assert float(ct_lookup(spec, jnp.asarray(3.0))) == CT_MAX
    assert np.all(np.asarray(spec.thrust_table)[3:] <= CT_MAX)


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
    ws = jnp.asarray(spec.wind_speed_table[16])
    expected = float(spec.power_table[16]) * POWER_SCALE
    assert float(power_lookup(spec, ws)) == pytest.approx(expected, rel=1e-9)


def test_power_lookup_fill_out_of_range_is_zero() -> None:
    spec = nrel5mw_v4()
    assert float(power_lookup(spec, jnp.asarray(-10.0))) == pytest.approx(0.0, abs=1e-9)
    assert float(power_lookup(spec, jnp.asarray(100.0))) == pytest.approx(0.0, abs=1e-9)
