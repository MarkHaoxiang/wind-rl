"""farm/ is fully implemented (design doc §"Package tree"); these run now."""

import math

import jax
import jax.numpy as jnp
import pytest

from windrl_engine.farm.layout import horns_rev2, row_layout
from windrl_engine.farm.turbine import (
    POWER,
    ROTOR_AREA,
    THRUST,
    WIND_SPEED,
    D,
    ct_interp,
    power_interp,
)
from windrl_engine.farm.turbine import (
    cp_interp as cp_interp_fn,
)
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


# --- Ct/Cp/inner-power interpolants (spec §5.5) -----------------------------


def test_ct_interp_matches_table_at_a_nonzero_node() -> None:
    # WIND_SPEED[20] = 11.5, THRUST[20] = 0.70701647 -- well inside the
    # (0.0001, 0.9999) clip range, so the post-clip is a no-op here.
    ws = WIND_SPEED[20]
    expected = THRUST[20]
    assert float(ct_interp(ws)) == pytest.approx(float(expected), abs=1e-9)


def test_ct_interp_clips_zero_valued_table_nodes_to_1e_minus_4() -> None:
    # THRUST[0] = THRUST[1] = 0.0 verbatim in the table, but ct_interp clips
    # every result (including exact table nodes) to (0.0001, 0.9999).
    assert float(ct_interp(WIND_SPEED[0])) == pytest.approx(0.0001, abs=1e-12)
    assert float(ct_interp(WIND_SPEED[1])) == pytest.approx(0.0001, abs=1e-12)


def test_cp_interp_is_exact_zero_at_a_zero_valued_node() -> None:
    # Unlike Ct, Cp has no post-clip (spec §5.5): fCp_interp fill is (0.0, 1.0)
    # and the table's own zero nodes pass through unclipped.
    assert float(cp_interp_fn(WIND_SPEED[0])) == pytest.approx(0.0, abs=1e-12)


def test_ct_interp_linear_midpoint_between_8_0_and_8_5() -> None:
    ws_mid = (WIND_SPEED[13] + WIND_SPEED[14]) / 2.0
    expected = (THRUST[13] + THRUST[14]) / 2.0
    assert float(ct_interp(ws_mid)) == pytest.approx(float(expected), abs=1e-9)


def test_cp_interp_linear_midpoint_between_8_0_and_8_5() -> None:
    ws_mid = (WIND_SPEED[13] + WIND_SPEED[14]) / 2.0
    expected = (POWER[13] + POWER[14]) / 2.0
    assert float(cp_interp_fn(ws_mid)) == pytest.approx(float(expected), abs=1e-9)


def test_power_interp_linear_midpoint_interpolates_inner_power_table() -> None:
    # power_interp interpolates the precomputed inner-power table
    # (0.5·rotor_area·Cp(ws)·gen_eff·ws³ per node, spec §5.5), NOT the physical
    # power recomputed at the midpoint wind speed -- so the expected value is
    # the average of the *table* endpoints, not 0.5·rotor_area·Cp(mid)·mid³.
    inner_power_13 = 0.5 * ROTOR_AREA * POWER[13] * WIND_SPEED[13] ** 3
    inner_power_14 = 0.5 * ROTOR_AREA * POWER[14] * WIND_SPEED[14] ** 3
    ws_mid = (WIND_SPEED[13] + WIND_SPEED[14]) / 2.0
    expected = (inner_power_13 + inner_power_14) / 2.0
    assert float(power_interp(ws_mid)) == pytest.approx(float(expected), rel=1e-9)


def test_ct_interp_fill_below_and_above_table_range() -> None:
    assert float(ct_interp(jnp.asarray(-10.0))) == pytest.approx(0.0001, abs=1e-12)
    assert float(ct_interp(jnp.asarray(100.0))) == pytest.approx(0.9999, abs=1e-12)


def test_cp_interp_fill_below_and_above_table_range() -> None:
    assert float(cp_interp_fn(jnp.asarray(-10.0))) == pytest.approx(0.0, abs=1e-12)
    assert float(cp_interp_fn(jnp.asarray(100.0))) == pytest.approx(1.0, abs=1e-12)


def test_power_interp_fill_below_and_above_table_range_is_zero() -> None:
    assert float(power_interp(jnp.asarray(-10.0))) == pytest.approx(0.0, abs=1e-9)
    assert float(power_interp(jnp.asarray(100.0))) == pytest.approx(0.0, abs=1e-9)
