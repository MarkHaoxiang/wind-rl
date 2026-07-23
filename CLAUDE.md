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

- **uv workspace**, Python 3.12 (Mava pins jax==0.5.3 / py<3.13, and the whole
  repo now shares one venv). The root `pyproject.toml` is a virtual workspace
  coordinator (tooling + dependency groups only, no `[project]` table).
  `packages/windrl-engine` is the JAX wind-farm simulator (a WFCRL/FLORIS GCH
  reimplementation, checked against committed FLORIS goldens); `packages/windrl-train`
  is the Mava MAPPO trainer plus the experiment harness (`config`/`settings`/`verdict`).
  WFCRL itself is no longer a dependency.
- All config objects are pydantic v2 (`extra="forbid"`) via the `Config` base
  in `packages/windrl-train/src/windrl_train/config.py` — a typo'd field is a
  `ValidationError`, not a silent no-op.
- **Commit frequently.** After any coherent set of related changes, commit
  without waiting to be asked. Never `git add -A` — stage specific files.
  Commit messages explain *why*, not what the diff already shows.

## Environment

One venv for the whole repo (py3.12). Mava is GitHub-only, its distribution is
named `id-mava`, and a non-editable wheel drops `mava/configs/`, so a PEP 508
git dependency can't work — it is installed **editable from a clone** at the
pinned SHA *after* `uv sync`, not declared in any `pyproject.toml`. Because it
isn't declared, a plain `uv sync` **prunes Mava's whole closure** (~150
packages). Setup and re-sync recipe (details in `packages/windrl-train/README.md`):

```bash
uv sync                              # first-time env
git clone https://github.com/instadeepai/Mava.git /tmp/mava && \
  git -C /tmp/mava checkout e1cc61dd0d3a5e02cab126cfb46ddcb7c32a5fdf
uv pip install -e /tmp/mava          # installs Mava + closure into .venv
# Later dependency changes: `uv sync --inexact` (preserves Mava). A plain
# `uv sync` prunes it — re-run the `uv pip install -e /tmp/mava` line if so.
```

## Checks

Run all four with `--no-sync` so the frozen Mava install (its own scipy/jax
pins) is never reconciled away:

```bash
uv run --no-sync ruff check packages/windrl-engine/src packages/windrl-engine/tests packages/windrl-train/src experiments
uv run --no-sync ruff format --check packages/windrl-engine/src packages/windrl-engine/tests packages/windrl-train/src experiments
uv run --no-sync mypy packages/windrl-engine/src packages/windrl-train/src   # + each experiments/*/ dir with .py files
uv run --no-sync pytest -q
```

`uv run pre-commit install` wires the first three (plus a fast pytest) into
a pre-commit hook; CI (`.github/workflows/ci.yml`) runs the same gate.

## CI and forbidden dependencies

The repo is torch-free (the DiCoDe `generative`/`design` torch code was
deleted with the old `wind-rl` package). CI (`.github/workflows/ci.yml`) is a
single job mirroring local setup: `uv sync` (py3.12), then clone + editable
`uv pip install -e` Mava at the pinned SHA, then ruff/format/mypy/pytest and
the import smoke + micro MAPPO run. The Mava install cannot be a PEP 508 git
dependency (its distribution is `id-mava`, and a non-editable wheel drops
`mava/configs/`, breaking Hydra's `pkg://mava.configs`), so it is always an
editable-from-clone step — keep that comment in the workflow.

The wandb pin is `>=0.19,<0.20` with an explicit `protobuf>=3.20,<4`
(`packages/windrl-train/pyproject.toml`): Mava imports `tensorboard_logger`
unconditionally, whose bundled pb2 stubs break on protobuf>=4, so the trainer
needs protobuf<4 to import at all; wandb>=0.20 ships pb2 stubs that require
protobuf>=5 at runtime, so 0.19 is the newest that co-resolves.

The FLORIS reference test (`packages/windrl-engine/tests/test_reference_solver.py`)
asserts against a single committed golden (`goldens/floris_v4.6.6.npz`), so no
test imports FLORIS and the whole suite runs on CI unfiltered — there is no `sim`
marker. FLORIS 4.6.6 is the sole reference: `generate_floris_goldens.py`
(isolated, `--with floris==4.6.6`) regenerates the solver golden, and
`generate_turbine_data.py` (same isolation) regenerates the shipped turbine
tables (`src/windrl_engine/farm/data/nrel5mw_v4.npz`) from FLORIS's packaged
`nrel_5MW.yaml`. Env semantics (invariants, duty-cycle, truncation) are checked
by golden-free tests (`test_invariants.py`, `test_env_pipeline.py`).

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
