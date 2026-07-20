"""Env-var pinning for FLORIS's numexpr/BLAS thread pools.

numexpr/OpenBLAS/MKL each read their thread-count env var once, at first
import in a process. Unpinned, every parallel FLORIS worker process defaults
to using all machine cores, so ``n_envs`` workers oversubscribe by a factor
of ``n_envs``. This only has a chance of taking effect for genuinely fresh
processes (``mp_start_method="spawn"``): a ``fork``-ed worker inherits the
parent's already-imported, already-configured numexpr/BLAS modules verbatim,
so mutating env vars around the fork is a no-op for those libraries.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

THREAD_ENV_VARS = (
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


@contextlib.contextmanager
def pinned_worker_threads(enabled: bool) -> Iterator[None]:
    """Force each of :data:`THREAD_ENV_VARS` to ``"1"`` for the wrapped block.

    Restores the previous value (or absence) of each var on exit, so the
    calling process's own threading is unaffected once the block ends. A
    no-op when ``enabled`` is ``False``.
    """
    if not enabled:
        yield
        return
    previous = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    os.environ.update(dict.fromkeys(THREAD_ENV_VARS, "1"))
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
