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

- **uv workspace**, Python 3.13. Root package is `src/wind_rl/` (src layout).
  `packages/wfcrl-env` is a **git submodule** (the author's WFCRL fork) —
  never edit it.
- All config objects are pydantic v2 (`extra="forbid"`) via the `Config` base
  in `src/wind_rl/config.py` — a typo'd field is a `ValidationError`, not a
  silent no-op.
- **Commit frequently.** After any coherent set of related changes, commit
  without waiting to be asked. Never `git add -A` — stage specific files.
  Commit messages explain *why*, not what the diff already shows.

## Checks

Before finishing any session, run all four:

```bash
uv run ruff check src tests experiments
uv run ruff format --check src tests
uv run mypy src tests
uv run pytest -q
```

`uv run pre-commit install` wires the first three (plus a fast pytest) into
a pre-commit hook; CI (`.github/workflows/ci.yml`) runs the same gate.

## torch, CI, and forbidden dependencies

torch/torchvision are pinned to the `pytorch-cu130` index (`[tool.uv.sources]`
in `pyproject.toml`). A full `uv sync` on CI would pull multi-GB CUDA wheels,
so CI installs dev tooling only (`uv sync --only-dev` + an editable, `--no-deps`
install of the project) and runs ruff/mypy/pytest against that. **This means
CI never imports torch and cannot catch a broken torch-dependent code path**
— treat GPU-touching changes (models, generative, rl) as needing a local run
before you trust them, not just green CI.

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
`WindRlSettings` (`src/wind_rl/experiment/settings.py`), overridable via
`WIND_RL_*` environment variables (e.g. `WIND_RL_WDIR`, `WIND_RL_WANDB_MODE`).
This is a direct fix for DiCoDe's hardcoded `~/.diffusion_co_design`.

## Parallel sessions

Multiple agents may work this repo at once. Never run `git reset --hard`,
`git checkout -- .`, or `git clean -fd` to tidy your own working tree — it can
destroy another session's uncommitted edits. Use `git stash` on your own paths
only, and commit green checkpoints promptly.
