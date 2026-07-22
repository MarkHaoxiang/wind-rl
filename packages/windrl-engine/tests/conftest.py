"""Turn the physics/farm modules' jaxtyping annotations into enforced runtime checks.

The single-farm core (``farm/*`` and ``physics/*``) is annotated with the
un-batched shape aliases from the design doc's fixed signatures (``"turbines"``,
``"turbines grid grid"``, ...). Installing jaxtyping's import hook (backed by
beartype) here makes those annotations *checked* during the test run -- every
call, and every ``jax.jit`` trace, verifies shapes and dtypes -- at zero cost to
the shipped package (the hook only exists in the test session).

The hook must be installed before the target modules are first imported, so
this lives in the top-level ``tests`` conftest, which pytest loads before any
test module.

``env`` / ``analysis`` are intentionally excluded: their per-turbine
annotations are the same un-batched aliases but execute batched under
``vmap`` (``BatchedWindFarmEnv``, rose evaluation), so enforcing them needs a
separate batched -> single-farm re-annotation pass first (see design doc
decision 3).
"""

import jax
from jaxtyping import install_import_hook

# The wake solve requires float64 for reference agreement (step spec §9); farm
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
    ],
    "beartype.beartype",
)
