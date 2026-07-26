# windrl-engine

A JAX reimplementation of the WFCRL/FLORIS GCH wake model. `farm/` describes a
site (layouts, turbine spec, wind), `physics/` solves the flow, `env/` wraps it
as an RL environment, `analysis/` and `viz/` inspect the result.

## Precision

The engine does **not** require float64.

- FLORIS parity is *defined* at float64: `tests/test_reference_solver.py`
  asserts the solve against a live FLORIS 4.6.6 run at `rtol=1e-9`.
- Under float32 the solve drifts to ~4e-7 relative on that same reference case.
  `tests/test_precision.py` pins both figures.
- Precision follows `jax_enable_x64` at solve time, not at import time: the
  turbine tables are numpy float64 and are canonicalised per trace.
- For training, float32 with `fidelity="corrected"` is the recommended
  configuration. `"corrected"` drops the reference's numerical quirks, including
  the ULP-sensitive wake gate that leaves `"floris"` non-rotation-invariant on
  some layouts.

## GPU

CPU JAX is the default. The NVIDIA CUDA 12 plugin is opt-in, so a plain
`uv sync` never pulls multi-GB wheels:

```bash
uv sync --extra gpu
```

## Testing

FLORIS 4.6.6 is a pinned runtime dependency, so the reference is computed live
in-process: there are no committed goldens and no generator scripts.
`test_reference_solver.py` drives FLORIS through its `"defaults"` GCH config
once per session, and `test_farm.py` asserts the turbine spec against the same
packaged `nrel_5MW.yaml` the engine itself reads. Upstream drift therefore
surfaces as a test failure rather than as silent divergence.
