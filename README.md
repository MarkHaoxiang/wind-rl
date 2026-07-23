# wind-rl

wind-rl scales multi-agent wind-farm layout co-design past DiCoDe's 16-turbine
setup (Li, Amir, Prorok, arXiv:2511.03100) toward real 32/64/92-turbine farms.

## Layout

```
pyproject.toml            # virtual uv workspace root: tooling + dependency groups only
packages/
  windrl-engine/          # pure-JAX wind-farm simulator (WFCRL/FLORIS GCH reimplementation)
    src/windrl_engine/    #   farm -> physics -> env (+ design, analysis)
  windrl-train/           # Mava MAPPO trainer + experiment harness
    src/windrl_train/
      env.py                #   Jumanji MarlEnv wrapper over the engine
      train.py              #   ff_mappo entrypoint + eval-series export
      networks.py           #   permutation-equivariant GCN torsos
      logging.py            #   wandb backend (WIND_RL_* contract)
      config.py             #   pydantic Config base (extra="forbid")
      settings.py           #   WindRlSettings (WIND_RL_* env vars)
      verdict.py            #   windowed_delta learning-signal score
experiments/              # numbered experiment frameworks (see experiments/README.md)
docs/
  architecture.md         # package/module map (kept current)
  research/               # research notes (owner-reviewed)
```

One uv workspace, one py3.12 venv. Mava pins `jax==0.5.3` / py<3.13, and is
installed editable from a clone (it can't be a declared dependency) — see
`packages/windrl-train/README.md`.

## Getting started

```bash
git clone --recurse-submodules <repo-url>
cd wind-rl
uv sync
# Mava is installed editable from a clone at a pinned SHA:
git clone https://github.com/instadeepai/Mava.git /tmp/mava-checkout
git -C /tmp/mava-checkout checkout e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf
uv pip install -e /tmp/mava-checkout
uv run --no-sync pre-commit install
```

Checks (also run in CI): see `CLAUDE.md`'s Checks section. A plain `uv sync`
prunes Mava — prefer `uv sync --inexact`, and run checks with `uv run --no-sync`.
