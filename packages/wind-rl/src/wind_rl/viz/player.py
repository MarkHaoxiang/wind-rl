"""Render a :class:`~wind_rl.viz.trajectory.ReplayTrajectory` to a standalone player.

:func:`build_replay_html` returns a single self-contained HTML document (vanilla
JS + canvas, no external resources) so it renders both as a local file and inside
wandb's iframe via ``wandb.Html(..., inject=False)``.
"""

from __future__ import annotations

from importlib.resources import files

from wind_rl.viz.trajectory import ReplayTrajectory

_PLACEHOLDER = "__TRAJECTORY_JSON__"


def load_template() -> str:
    """The player HTML template shipped alongside this module."""
    return (
        files("wind_rl.viz")
        .joinpath("replay_template.html")
        .read_text(encoding="utf-8")
    )


def build_replay_html(traj: ReplayTrajectory) -> str:
    """Embed ``traj`` into the player template, yielding one standalone HTML string."""
    # Escape ``<`` so a stray "</script>" inside the payload cannot close the
    # inline script early; the JSON stays valid (< is a legal escape).
    payload = traj.model_dump_json().replace("<", "\\u003c")
    template = load_template()
    if _PLACEHOLDER not in template:  # pragma: no cover - template integrity guard
        raise ValueError(f"replay template is missing {_PLACEHOLDER!r}")
    return template.replace(_PLACEHOLDER, payload)
