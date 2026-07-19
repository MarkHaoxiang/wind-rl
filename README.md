# wind-rl

wind-rl scales multi-agent wind-farm layout co-design past DiCoDe's 16-turbine
setup (Li, Amir, Prorok, arXiv:2511.03100) toward real 32/64/92-turbine farms.

## Layout

```
pyproject.toml          # virtual uv workspace root: tooling + dependency groups only
packages/
  wind-rl/              # the main package (its own pyproject.toml, uv_build backend)
    src/wind_rl/        # the wind_rl package (src layout)
      config.py            # pydantic Config base + OmegaConf/Hydra override merge
      scenario.py          # ScenarioConfig + real-farm registry
      experiment/settings.py  # WindRlSettings (WIND_RL_* env vars)
      env/                 # WFCRL env wrapper, factory, transforms
      models/              # policy/critic and generator architectures
      generative/           # flow-map / diffusion layout generators
      design/               # Designer abstraction, layout buffer, baseline designers
      rl/                   # MAPPO trainer
    tests/              # pytest suite, mirrors src/wind_rl/
  wfcrl-env/            # git submodule: the author's WFCRL fork, consumed as a library
experiments/           # numbered experiment frameworks (see experiments/README.md)
docs/
  plans/               # architecture & research plans (owner-reviewed)
  research/            # research notes (owner-reviewed)
```

## Getting started

```bash
git clone --recurse-submodules <repo-url>
cd wind-rl
uv sync
uv run pre-commit install
```

Checks (also run in CI): see `CLAUDE.md`'s Checks section.

## Documentation

See `docs/plans/2026-07-19-wind-rl-architecture.md` for the project's
architecture plan, milestone roadmap, and owner decisions.
