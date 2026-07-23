"""Install jaxtyping/beartype runtime shape checks for the farm/physics single-farm core.

The un-batched shape aliases (``"turbines"``, ``"turbines grid grid"``, ...) on
``farm/*`` and ``physics/*`` become *checked* for the test run -- every call and
every ``jax.jit`` trace verifies shapes and dtypes -- at zero cost to the
shipped package (the hook only exists in the test session). It must be
installed before those modules are first imported, so this lives in the
top-level conftest, which pytest loads before any test module.

``env`` / ``analysis`` are excluded: they execute the same un-batched aliases
batched under ``vmap`` (``BatchedWindFarmEnv``, rose evaluation), which needs a
separate batched -> single-farm re-annotation pass first.
"""

import jax
from jaxtyping import install_import_hook

# The wake solve requires float64 for reference agreement; farm
# tables and layouts are built at import time, so this must run before any of
# the hooked modules (or their dependents) are first imported.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

install_import_hook(
    [
        "windrl_engine.farm.layout",
        "windrl_engine.farm.state",
        "windrl_engine.farm.turbine",
        "windrl_engine.farm.wind",
        "windrl_engine.physics.frame",
        "windrl_engine.physics.flow",
        "windrl_engine.physics.thrust",
        "windrl_engine.physics.deflection",
        "windrl_engine.physics.transverse",
        "windrl_engine.physics.deficit",
        "windrl_engine.physics.turbulence",
        "windrl_engine.physics.solver",
        "windrl_engine.physics.power",
        "windrl_engine.design.base",
        "windrl_engine.design.designers",
        "windrl_engine.design.feasibility",
    ],
    "beartype.beartype",
)
