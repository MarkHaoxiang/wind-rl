"""Designer layer tests: feasibility predicates, projection, and baseline designers."""

import jax
import jax.numpy as jnp
import pytest

from windrl_engine.design import (
    fixed,
    in_bounds,
    make_site,
    min_spacing_satisfied,
    project_feasible,
    random_uniform,
)
from windrl_engine.farm.layout import FarmLayout


def _min_off_diagonal_distance(coords: jax.Array) -> jax.Array:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    n = coords.shape[0]
    return jnp.min(jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist))


@pytest.mark.parametrize("n_turbines", [7, 32])
def test_random_uniform_batches_are_entirely_feasible(n_turbines: int) -> None:
    # A loose site (4 km square, 250 m minimum) so a correct projection has ample
    # room to separate every proposal; feasibility is asserted as a 100% invariant.
    site = make_site(4000.0, 4000.0, 250.0)
    batch = 256
    layouts = random_uniform(site, n_turbines)(jax.random.key(n_turbines), batch)

    assert layouts.shape == (batch, n_turbines, 2)
    spacing_ok = jax.vmap(lambda c: min_spacing_satisfied(c, site.min_spacing))(layouts)
    bounds_ok = jax.vmap(lambda c: in_bounds(c, site))(layouts)
    assert bool(jnp.all(spacing_ok))
    assert bool(jnp.all(bounds_ok))


def test_random_uniform_is_deterministic_per_key_and_differs_across_keys() -> None:
    site = make_site(3000.0, 3000.0, 200.0)
    designer = random_uniform(site, 12)

    same_a = designer(jax.random.key(0), 8)
    same_b = designer(jax.random.key(0), 8)
    different = designer(jax.random.key(1), 8)

    assert bool(jnp.array_equal(same_a, same_b))
    assert not bool(jnp.allclose(same_a, different))


def test_fixed_tiles_its_layout_exactly_and_ignores_the_key() -> None:
    layout = FarmLayout(
        x=jnp.asarray([0.0, 504.0, 1008.0]),
        y=jnp.asarray([0.0, 120.0, -60.0]),
    )
    designer = fixed(layout)
    expected_tile = jnp.stack([layout.x, layout.y], axis=-1)  # (3, 2)

    out = designer(jax.random.key(4), 5)
    assert out.shape == (5, 3, 2)
    for lane in range(5):
        assert bool(jnp.array_equal(out[lane], expected_tile))
    # Deterministic: a different key tiles the identical layout.
    assert bool(jnp.array_equal(out, designer(jax.random.key(99), 5)))


def test_project_feasible_repairs_a_violating_cluster_within_bounds() -> None:
    site = make_site(2000.0, 2000.0, 500.0)
    # First two turbines are 200 m apart -- below the 500 m minimum.
    coords = jnp.asarray([[100.0, 100.0], [300.0, 100.0], [1500.0, 1500.0]])
    assert not bool(min_spacing_satisfied(coords, site.min_spacing))

    projected = project_feasible(coords, site)

    assert projected.shape == coords.shape
    assert bool(min_spacing_satisfied(projected, site.min_spacing))
    assert bool(in_bounds(projected, site))
    # The repaired minimum separation reaches (to numerical tolerance) the target.
    assert float(_min_off_diagonal_distance(projected)) >= 500.0 - 1e-6


def test_feasibility_predicates_agree_with_hand_computed_distances() -> None:
    # 3-4-5 right triangle: pairwise distances AB=3, AC=4, BC=5, so min = 3.
    coords = jnp.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    assert float(_min_off_diagonal_distance(coords)) == pytest.approx(3.0)

    assert bool(min_spacing_satisfied(coords, jnp.asarray(2.5)))
    assert not bool(min_spacing_satisfied(coords, jnp.asarray(3.5)))

    site = make_site(10.0, 10.0, 1.0)
    assert bool(in_bounds(coords, site))
    beyond_extent = jnp.asarray([[0.0, 0.0], [11.0, 0.0], [0.0, 4.0]])
    below_origin = jnp.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, -1.0]])
    assert not bool(in_bounds(beyond_extent, site))
    assert not bool(in_bounds(below_origin, site))
