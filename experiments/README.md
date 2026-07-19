# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class
of related runs sharing machinery, not a single run. Prefer adding a variant
to an existing framework over scaffolding a new number.

## Layout

```
experiments/
├── README.md              # this contract
├── JOURNAL.md              # one verdict line per concluded finding, append-only
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

Runs never write into the repo. All local artifacts (checkpoints, logged
layouts, rendered videos, config snapshots) go under `WIND_RL_WDIR`
(`WindRlSettings`, default `~/.wind_rl`, override via the `WIND_RL_WDIR` env
var) — set it per-machine, not hardcoded in a script.

Tracking (config, metrics, media, verdict) goes to **Weights & Biases**,
controlled by `WIND_RL_WANDB_MODE` / `WANDB_MODE`:

```bash
WANDB_MODE=disabled uv run python experiments/NNNN_slug/run.py   # fast plumbing check, no tracking
WANDB_MODE=offline  uv run python experiments/NNNN_slug/run.py   # record locally, `wandb sync` later
WANDB_MODE=online   uv run python experiments/NNNN_slug/run.py   # requires `wandb login`
```

## JOURNAL.md

Append-only. One line per concluded finding: framework, one-line verdict,
pointer to the run/report. Don't journal in package docs (`CLAUDE.md` cites
experiment numbers instead) and don't rewrite past entries.

## Planned frameworks

See `docs/plans/2026-07-19-wind-rl-architecture.md` (milestone roadmap) and
`docs/research/2026-07-19-geometric-architectures.md` (staged architecture
path and promotion gates) for what's planned.
