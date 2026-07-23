# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class
of related runs sharing machinery, not a single run. Prefer adding a variant
to an existing framework over scaffolding a new number.

**A new study is a new `conf/` variant set in an existing framework, not a new
directory — unless the experiment SHAPE is genuinely new.** All shared
machinery (variant sweeps, comparison tables, verdict gates, wandb handling,
architecture-benchmark proxies) lives in `wind_rl.experiment`; a framework is
only a `conf/`, a thin `run.py` of glue, and a `report.md`.

## Layout

```
experiments/
├── README.md              # this contract
├── JOURNAL.md              # owner-managed verdict log (see below)
└── NNNN_slug/
    ├── run.py               # entry point
    ├── conf/                # config groups (variants)
    └── report.md            # Hypothesis -> Setup -> Results -> Decision
```

## Anatomy of a framework

- **`run.py`** builds a typed `Config` (the `wind_rl.config.Config` base,
  `extra="forbid"`) from `conf/`, then drives the run: collect ->
  designer/policy update -> periodic eval -> checkpoint.
- **`conf/`** holds config groups; select a variant explicitly rather than
  patching top-level keys ad hoc.
- **`report.md`** records the finding: **Hypothesis** (what claim is being
  tested, tie back to the plan's C1-C4 where relevant) -> **Setup** (scenario,
  designer/architecture, seeds, budget) -> **Results** (numbers, pointers to
  the run) -> **Decision** (what this settles, and what runs next).

## Verdicts are asserted in code, never by eye

Every framework gates its claim on a threshold checked in code (e.g. "power
strictly increases", ">=10x fewer NFEs", "beats v1 by margin M") and passes
the resulting verdict through the run's finish/summary step. A report without
an asserted threshold is a plot, not a finding.

## Where outputs land

All local artifacts (checkpoints, logged layouts, rendered videos, config
snapshots) go under `WIND_RL_WDIR` (`WindRlSettings`, default `outputs/` at the
repo root — gitignored, so runs never dirty the tree; override via the
`WIND_RL_WDIR` env var) — set it per-machine, not hardcoded in a script.

Tracking (config, metrics, media, verdict) goes to **Weights & Biases**,
controlled by `WIND_RL_WANDB_MODE` (plain `WANDB_MODE` is not honoured —
`WindRlSettings` reads only `WIND_RL_*` variables):

```bash
WIND_RL_WANDB_MODE=disabled uv run python experiments/NNNN_slug/run.py   # fast plumbing check, no tracking
WIND_RL_WANDB_MODE=offline  uv run python experiments/NNNN_slug/run.py   # record locally, `wandb sync` later
WIND_RL_WANDB_MODE=online   uv run python experiments/NNNN_slug/run.py   # requires `wandb login`
```

## JOURNAL.md

**Owner-managed.** The owner writes journal entries, or dictates them
verbatim; agents never append on their own initiative. One line per concluded
finding: framework, one-line verdict, pointer to the run/report. Don't journal
in package docs (`CLAUDE.md` cites experiment numbers instead) and don't
rewrite past entries.

## Planned frameworks

See `docs/plans/2026-07-19-wind-rl-architecture.md` (milestone roadmap) and
`docs/research/2026-07-19-geometric-architectures.md` (staged architecture
path and promotion gates) for what's planned.
