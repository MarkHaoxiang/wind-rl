"""Install jaxtyping/beartype runtime shape checks across the engine.

The shape aliases (``"turbines"``, ``"turbines grid grid"``, ``"envs
turbines"``, ...) on the hooked modules become *checked* for the test run --
every call and every ``jax.jit`` trace verifies shapes and dtypes -- at zero
cost to the shipped package (the hook only exists in the test session). It must
be installed before those modules are first imported, so this lives in the
top-level conftest, which pytest loads before any test module.

Coverage is every module of ``farm/``, ``physics/``, ``design/`` and ``env/``,
plus ``metrics.py``, ``viz/plane.py`` and ``viz/record.py``. Two engine modules
are excluded: ``viz/field.py``, whose ``npt.NDArray[np.float32]`` returns
beartype rejects outright at decoration time
(``BeartypeDecorHintNonpepNumpyException``), and ``viz/server.py``, which
annotates no arrays at all.
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
        "windrl_engine.physics.query_field",
        "windrl_engine.physics.power",
        "windrl_engine.design.base",
        "windrl_engine.design.designers",
        "windrl_engine.design.feasibility",
        "windrl_engine.metrics",
        "windrl_engine.env.spaces",
        "windrl_engine.env.actions",
        "windrl_engine.env.config",
        "windrl_engine.env.reward",
        "windrl_engine.env.single_farm",
        "windrl_engine.env.batched",
        "windrl_engine.viz.plane",
        "windrl_engine.viz.record",
    ],
    "beartype.beartype",
)
