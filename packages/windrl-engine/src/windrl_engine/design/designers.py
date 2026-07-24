"""Static / search-free baseline designers.

`fixed` tiles a concrete `FarmLayout` over the batch and doubles as the "manual"
published-layout designer: `fixed(horns_rev2())` is the published anchor (see
`farm.layout` for the reference builders). `random_uniform` draws uniform
proposals and projects them onto the feasible set. Learning/search designers
(Sampling/Descent/Reinforce/Replay) require a critic/objective and are a later
slice — deliberately absent here.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from windrl_engine.design.base import Designer
from windrl_engine.design.feasibility import SiteSpec, project_feasible
from windrl_engine.farm.layout import FarmLayout


def fixed(layout: FarmLayout) -> Designer:
    coords = jnp.stack([layout.x, layout.y], axis=-1)  # (turbines, 2)

    def designer(
        key: Key[Array, ""], batch_size: int
    ) -> Float[Array, "batch turbines 2"]:
        del key  # deterministic: the layout is tiled unchanged
        return jnp.broadcast_to(coords, (batch_size, *coords.shape))

    return designer


def random_uniform(site: SiteSpec, n_turbines: int, iters: int = 200) -> Designer:
    def designer(
        key: Key[Array, ""], batch_size: int
    ) -> Float[Array, "batch turbines 2"]:
        maxval = jnp.stack([site.x_extent, site.y_extent])
        proposals = jax.random.uniform(
            key, (batch_size, n_turbines, 2), minval=0.0, maxval=maxval
        )
        return jax.vmap(lambda c: project_feasible(c, site, iters))(proposals)

    return designer
