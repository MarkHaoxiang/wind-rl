"""Site feasibility and a jit-safe iterative projection onto the feasible set.

The feasible set is the intersection of a rectangular boundary
(``[0, x_extent] x [0, y_extent]``) and a pairwise min-spacing constraint.
Only the rectangle is implemented today; ``SiteSpec`` and the ``site``-taking
functions (:func:`in_bounds`, :func:`project_feasible`) are the seam through
which a polygon boundary would later be threaded.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class SiteSpec(NamedTuple):
    x_extent: Float[Array, ""]  # meters; usable region is [0, x_extent]
    y_extent: Float[Array, ""]  # meters; usable region is [0, y_extent]
    min_spacing: Float[Array, ""]  # meters, pairwise turbine-turbine minimum


def make_site(x_extent: float, y_extent: float, min_spacing: float) -> SiteSpec:
    return SiteSpec(
        x_extent=jnp.asarray(x_extent),
        y_extent=jnp.asarray(y_extent),
        min_spacing=jnp.asarray(min_spacing),
    )


def _pairwise_distance(
    coords: Float[Array, "turbines 2"],
) -> Float[Array, "turbines turbines"]:
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def min_spacing_satisfied(
    coords: Float[Array, "turbines 2"], min_spacing: Float[Array, ""]
) -> Bool[Array, ""]:
    n = coords.shape[0]
    dist = _pairwise_distance(coords)
    off_diagonal = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist)
    return jnp.all(off_diagonal >= min_spacing)


def in_bounds(coords: Float[Array, "turbines 2"], site: SiteSpec) -> Bool[Array, ""]:
    x, y = coords[:, 0], coords[:, 1]
    return jnp.all(
        (x >= 0.0) & (x <= site.x_extent) & (y >= 0.0) & (y <= site.y_extent)
    )


def project_feasible(
    coords: Float[Array, "turbines 2"],
    site: SiteSpec,
    iters: int = 200,
    tol: float = 1e-3,
) -> Float[Array, "turbines 2"]:
    """Push `coords` toward the feasible set: pairwise repulsion then bounds clip.

    Each iteration moves every turbine half of every min-spacing violation along
    the separating axis, then clips to the rectangle. Repulsion targets
    ``min_spacing * (1 + tol)`` rather than ``min_spacing`` exactly: a bare
    target settles pairs on the constraint boundary where floating-point rounding
    lands them a few ULPs *below* it and fails the strict ``>=`` feasibility
    check, so the small overshoot is what makes convergence robust. Convergence
    caveat: the move is a fixed-count local relaxation, not an exact projection —
    summed repulsion from many neighbours can overshoot and a corner clip can
    reintroduce a violation, so a large enough `iters` (and a genuinely
    satisfiable site) is the caller's responsibility. Fully jit/vmap-safe: dense
    ``(N, N)`` math, no ragged ops, static iteration structure.
    """
    n = coords.shape[0]
    not_self = ~jnp.eye(n, dtype=bool)
    target = site.min_spacing * (1.0 + tol)

    def step(_: Array, c: Float[Array, "turbines 2"]) -> Float[Array, "turbines 2"]:
        diff = c[:, None, :] - c[None, :, :]  # (N, N, 2), pos_i - pos_j
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (N, N)
        safe_dist = jnp.where(dist > 0.0, dist, 1.0)
        unit = diff / safe_dist[..., None]  # separating direction, j -> i
        overlap = jnp.where(not_self & (dist < target), target - dist, 0.0)
        displacement = jnp.sum(unit * (0.5 * overlap)[..., None], axis=1)  # (N, 2)
        moved = c + displacement
        x = jnp.clip(moved[:, 0], 0.0, site.x_extent)
        y = jnp.clip(moved[:, 1], 0.0, site.y_extent)
        return jnp.stack([x, y], axis=-1)

    projected: Float[Array, "turbines 2"] = jax.lax.fori_loop(0, iters, step, coords)
    return projected
