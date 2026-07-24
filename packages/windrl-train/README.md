# windrl-train

The experiment harness shared with `experiments/`: the pydantic config base,
the `WIND_RL_*` settings contract, the verdict scoring, and a wandb logger. The
RL trainer over `windrl-engine` is being rewritten in-repo (the Mava clone at
`/home/markhaoxiang/Projects/mava` is kept purely as a reading reference).

- `config.py` — `Config`, the pydantic v2 `extra="forbid"` base every typed
  experiment config extends, with OmegaConf-backed YAML + dotlist loading.
- `settings.py` — `WindRlSettings`, the one `WIND_RL_*` env-var contract in the
  repo (`WIND_RL_WDIR`, `WIND_RL_WANDB_MODE`).
- `verdict.py` — `windowed_delta`, the per-run learning-signal score frameworks
  gate on.
- `logging.py` — `WandbLogger`, a wandb run wrapper honoring the settings
  contract. The run `name` deliberately omits the seed so seeds share a name and
  wandb group-by-name shows the seed distribution; the seed goes in `tags`.

## Setup

From the repo root, `uv sync` builds the whole-repo venv (both workspace
members and their declared closures). See the repo `CLAUDE.md` for the checks
gate.

## GPU (NVIDIA CUDA 12)

GPU support is opt-in and lives on `windrl-engine` (jax is its dependency):

```bash
uv sync --extra gpu
```

This layers the `jax[cuda12]` plugin wheels on top of the CPU jax. A default
`uv sync` (no `--extra`) omits them, so CI never pulls the multi-GB CUDA wheels.
The bundled wheels cover Blackwell (`sm_120`, e.g. RTX 5090) natively — a
current NVIDIA driver is the only host requirement; no system CUDA toolkit is
needed. Verify:

```bash
uv run python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

With the plugin installed jax defaults to the GPU; force CPU with
`JAX_PLATFORMS=cpu`. The engine's FLORIS golden parity tests pass on GPU at the
same tolerance as CPU.
