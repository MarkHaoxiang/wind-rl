"""Leaf module of pure constants with no non-stdlib imports.

Anything here must stay importable without wfcrl/torch/etc. installed, so
consumers (e.g. :mod:`wind_rl.models`) can be imported standalone.
"""

from __future__ import annotations

#: Name of the single agent group (all turbines share one policy/critic).
GROUP_NAME = "turbine"
#: Name of the environment family.
ENV_NAME = "wfcrl"
