# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

wind-rl scales wind-farm layout co-design (equivariant architectures, per-site
fine-tuning) past DiCoDe (arXiv:2511.03100) to real 32/64/92-turbine farms.
`docs/architecture.md` is the high-level package map — **keep it updated**
whenever the package layout or module responsibilities change (one sentence
max per module).

## Conventions

Coding style is governed by [docs/coding-guidelines.md](docs/coding-guidelines.md):
clear abstractions, strict typing IS the documentation, comments minimal and
only on user-facing API or genuine *why*.

- **uv workspace**, Python 3.12, one venv for the whole repo. The root
  `pyproject.toml` is a virtual workspace coordinator (tooling + dependency
  groups only, no `[project]` table). `packages/windrl-engine` is the JAX
  wind-farm simulator (a WFCRL/FLORIS GCH reimplementation, checked live against
  FLORIS 4.6.6); `packages/windrl-train` is the experiment harness
  (`config`/`settings`/`verdict`/wandb logging) — the RL trainer is being
  rewritten in-repo. WFCRL itself is no longer a dependency. A read-only Mava
  clone at `/home/markhaoxiang/Projects/mava` is kept purely as a reading
  reference for that rewrite.
- All config objects are pydantic v2 (`extra="forbid"`) via the `Config` base
  in `packages/windrl-train/src/windrl_train/config.py` — a typo'd field is a
  `ValidationError`, not a silent no-op.
- **Commit frequently.** After any coherent set of related changes, commit
  without waiting to be asked. Never `git add -A` — stage specific files.
  Commit messages explain *why*, not what the diff already shows.

## Environment

One venv for the whole repo (py3.12), fully declared in the two package
`pyproject.toml`s — a plain `uv sync` builds it, and `uv.lock` is committed:

```bash
uv sync                # first-time env (CPU)
uv sync --extra gpu    # add the NVIDIA CUDA 12 jax plugin (opt-in, see engine README)
```

## Checks

```bash
uv run ruff check packages/windrl-engine/src packages/windrl-engine/tests packages/windrl-train/src experiments
uv run ruff format --check packages/windrl-engine/src packages/windrl-engine/tests packages/windrl-train/src experiments
uv run mypy packages/windrl-engine/src packages/windrl-train/src   # + each experiments/*/ dir with .py files
uv run pytest -q
```

`uv run pre-commit install` wires the first three (plus a fast pytest) into
a pre-commit hook; CI (`.github/workflows/ci.yml`) runs the same gate.

## CI and forbidden dependencies

The repo is torch-free (the DiCoDe `generative`/`design` torch code was
deleted with the old `wind-rl` package). CI (`.github/workflows/ci.yml`) is a
single job mirroring local setup: `uv sync` (py3.12), then
ruff/format/mypy/pytest.

FLORIS 4.6.6 is a pinned runtime dependency of windrl-engine, so the reference
is computed live in-process — there are no committed goldens and no isolated
generator scripts. The turbine tables are read from FLORIS's packaged
`nrel_5MW.yaml` at import (`farm/turbine.py`, without importing floris itself),
and `test_reference_solver.py` runs FLORIS through its `"defaults"`
GCH config once per session (module fixture, CPU) and asserts the JAX solve
against it at rtol 1e-12 for u/turbulence-intensity/power and a looser rtol
1e-9 (+atol) for the near-zero v/w transverse components. `test_farm.py`
asserts `turbine.py` matches that same packaged YAML, so upstream drift is
caught without a frozen file. Env semantics
(invariants, duty-cycle, truncation) are checked by golden-free tests
(`test_invariants.py`, `test_env_pipeline.py`).

**If torch is reintroduced (e.g. for the `generative`/`design` rewrite), never
add `torch_scatter` or `torch_cluster`.** DiCoDe's manual
`--no-build-isolation` wheel-build pain for these is exactly what this project
avoids. Architectures must stay torch-native: dense adjacency / `index_add_`
message passing and `torch.cdist`/`topk` for KNN graphs are cheap enough at
N<=92 turbines. `torch-geometric` alone (no scatter/cluster wheels) is the only
sanctioned escape hatch, and only if DeepSets/EGNN-style dense layers turn out
insufficient.

## Experiments

ML experiments live in `experiments/` (contract: `experiments/README.md`).
Numbered `NNNN_slug/` directories are frameworks, not single runs; verdicts
are asserted in code. `experiments/JOURNAL.md` is **owner-managed**: the owner
writes entries or dictates them verbatim — never append to it unprompted.

## Plans are temporary; docs/research is owner-reviewed

Plan documents are temporary objects: once the work they describe is complete,
delete the plan (and any references to it) rather than archiving it — current
state belongs in code, tests, and `docs/architecture.md`. `docs/research` is
the owner's curated research record: propose changes (new findings, scope
shifts, corrected assumptions) rather than silently rewriting, and let the
owner confirm before editing the file itself.

## No hardcoded paths

Never hardcode a working directory, checkpoint path, or wandb setting. Use
`WindRlSettings` (`packages/windrl-train/src/windrl_train/settings.py`), overridable via
`WIND_RL_*` environment variables (e.g. `WIND_RL_WDIR`, `WIND_RL_WANDB_MODE`).
This is a direct fix for DiCoDe's hardcoded `~/.diffusion_co_design`.

## Parallel sessions

Multiple agents may work this repo at once. Never run `git reset --hard`,
`git checkout -- .`, or `git clean -fd` to tidy your own working tree — it can
destroy another session's uncommitted edits. Use `git stash` on your own paths
only, and commit green checkpoints promptly.
