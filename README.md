# wind-rl

wind-rl scales multi-agent wind-farm layout co-design past DiCoDe's 16-turbine
setup (Li, Amir, Prorok, arXiv:2511.03100) toward real 32/64/92-turbine farms.

## Layout

```
pyproject.toml            # virtual uv workspace root: tooling + dependency groups only
packages/
  windrl-engine/          # pure-JAX wind-farm simulator (WFCRL/FLORIS GCH reimplementation)
    src/windrl_engine/    #   farm -> physics -> env, viz above both (+ design, metrics)
  windrl-train/           # experiment harness (RL trainer being rewritten in-repo)
    src/windrl_train/
      config.py             #   pydantic Config base (extra="forbid")
      settings.py           #   WindRlSettings (WIND_RL_* env vars)
      verdict.py            #   windowed_delta learning-signal score
      logging.py            #   wandb logger (WIND_RL_* contract)
experiments/              # numbered experiment frameworks (see experiments/README.md)
docs/
  architecture.md         # package/module map (kept current)
  research/               # research notes (owner-reviewed)
```

One uv workspace, one py3.12 venv, fully declared in the package
`pyproject.toml`s. Add the NVIDIA CUDA 12 jax plugin with `uv sync --extra gpu`
(see `packages/windrl-engine`).

## Getting started

```bash
git clone <repo-url>
cd wind-rl
uv sync
uv run pre-commit install
```

Checks (also run in CI): see `CLAUDE.md`'s Checks section.
