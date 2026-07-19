# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

wind-rl scales wind-farm layout co-design (flow-map generators, equivariant
architectures, per-site fine-tuning) past DiCoDe (arXiv:2511.03100) to real
32/64/92-turbine farms. Read `docs/plans/2026-07-19-wind-rl-architecture.md`
before working on architecture, milestones, or scope — it is the source of
truth for the code layout, interfaces, and the owner decisions in its §5.

## Conventions

Coding style is governed by [docs/coding-guidelines.md](docs/coding-guidelines.md):
clear abstractions, strict typing IS the documentation, comments minimal and
only on user-facing API or genuine *why*.

- **uv workspace**, Python 3.13. The root `pyproject.toml` is a virtual
  workspace coordinator (tooling + dependency groups only, no `[project]`
  table); the main package lives at `packages/wind-rl/src/wind_rl/` (src
  layout, own `pyproject.toml`). `packages/wfcrl-env` is a **git submodule**
  (the author's WFCRL fork) — never edit it.
- All config objects are pydantic v2 (`extra="forbid"`) via the `Config` base
  in `packages/wind-rl/src/wind_rl/config.py` — a typo'd field is a
  `ValidationError`, not a silent no-op.
- **Commit frequently.** After any coherent set of related changes, commit
  without waiting to be asked. Never `git add -A` — stage specific files.
  Commit messages explain *why*, not what the diff already shows.

## Checks

Before finishing any session, run all four:

```bash
uv run ruff check packages/wind-rl/src packages/wind-rl/tests experiments
uv run ruff format --check packages/wind-rl/src packages/wind-rl/tests
uv run mypy packages/wind-rl/src packages/wind-rl/tests
uv run pytest -q
```

`uv run pre-commit install` wires the first three (plus a fast pytest) into
a pre-commit hook; CI (`.github/workflows/ci.yml`) runs the same gate.

## torch, CI, and forbidden dependencies

torch/torchvision are pinned to the `pytorch-cu130` index (`[tool.uv.sources]`
in `pyproject.toml`). A full `uv sync` on CI would pull multi-GB CUDA wheels, so
CI installs sequentially instead (`.github/workflows/ci.yml`): `uv sync
--only-dev`, then a **CPU** torch (`--index-url .../whl/cpu`), then `torchrl` and
the pure-Python runtime deps, then an editable `--no-deps` install of
`packages/wind-rl`. This leaves local resolution untouched (no `pyproject`/
`uv.lock` change) while giving CI a real torch + numpy so **`mypy
packages/wind-rl/{src,tests}` (strict) and `pytest` genuinely run** — CI now
catches torch-dependent breakage. Caveat: the CPU index may serve a
torch/tensordict slightly newer than the locked CUDA pin, so CI validates a
near-but-not-identical stack; the locked stack runs locally.

Root `dependency-groups.dev` deliberately does **not** list `wind-rl` as a
workspace member (unlike a plain catan-engine-style copy): a virtual
workspace's plain `uv sync` already installs every `[tool.uv.workspace]`
member regardless of dev-group membership, so `import wind_rl` works at root
without it. Listing it would make CI's `uv sync --only-dev` step also pull
wind-rl's full dependency closure — including cu130 torch — defeating the
staged CPU-torch install above.

CI's pytest is scoped to `-m "not sim"`. Tests marked `sim` import `wfcrl` (the
workspace member), which needs system MPI + FLORIS the runner lacks; they are
excluded from CI but still run locally and in the pre-commit hook (which runs
the full `pytest -q`). Mark any new FLORIS/wfcrl-touching test `sim` and guard
its module with `pytest.importorskip("wfcrl")` so CI collection stays green.

**Never add `torch_scatter` or `torch_cluster`.** DiCoDe's manual
`--no-build-isolation` wheel-build pain for these is exactly what this project
avoids. Architectures must stay torch-native: dense adjacency / `index_add_`
message passing and `torch.cdist`/`topk` for KNN graphs are cheap enough at
N<=92 turbines. See the plan's §3 (Dependency plan) before adding any geometric
dependency — `torch-geometric` alone (no scatter/cluster wheels) is the only
sanctioned escape hatch, and only if DeepSets/EGNN-style dense layers turn out
insufficient.

## Experiments

ML experiments live in `experiments/` (contract: `experiments/README.md`).
Numbered `NNNN_slug/` directories are frameworks, not single runs; verdicts
are asserted in code and journalled to `experiments/JOURNAL.md`.

## docs/plans and docs/research are owner-reviewed

Both are the owner's curated planning and research record. Propose changes
(new findings, scope shifts, corrected assumptions) rather than silently
rewriting them — flag what changed and why, and let the owner confirm before
editing the file itself.

## No hardcoded paths

Never hardcode a working directory, checkpoint path, or wandb setting. Use
`WindRlSettings` (`packages/wind-rl/src/wind_rl/experiment/settings.py`), overridable via
`WIND_RL_*` environment variables (e.g. `WIND_RL_WDIR`, `WIND_RL_WANDB_MODE`).
This is a direct fix for DiCoDe's hardcoded `~/.diffusion_co_design`.

## Parallel sessions

Multiple agents may work this repo at once. Never run `git reset --hard`,
`git checkout -- .`, or `git clean -fd` to tidy your own working tree — it can
destroy another session's uncommitted edits. Use `git stash` on your own paths
only, and commit green checkpoints promptly.
