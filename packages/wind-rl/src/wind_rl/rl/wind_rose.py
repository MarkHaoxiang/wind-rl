"""Wind-rose histogram and the eval-time rose configuration (WFCRL Scenario II).

The WFCRL benchmark's windrose scenario trains under freely-sampled free-stream
wind and scores a policy against a *wind rose*: a direction x speed frequency
histogram of a real measurement campaign. Eval runs one deterministic episode per
histogram bin (wind fixed to the bin representative) and reports the
frequency-weighted mean episode return.

:func:`prepare_wind_rose` is the pure, unit-tested histogram builder (bins + freqs
from raw direction/speed arrays). :class:`WindRose` is its numpy-side value type;
:class:`WindRoseEvalConfig` is the pydantic knob that carries a precomputed rose
through :class:`~wind_rl.rl.trainer.TrainingConfig` into the trainer's eval path.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from wind_rl.config import Config

#: SMARTEOLE wind directions are stored in a turbine-row-relative frame; the WFCRL
#: benchmark rotates them by +60 deg into the FLORIS absolute frame before binning
#: (see the reference ``prepare_eval_windrose``). Applied to the *data*, so the
#: resulting bin edges/centers are already absolute wind directions.
SMARTEOLE_DIRECTION_OFFSET = 60.0


@dataclass(frozen=True)
class WindRose:
    """A direction x speed free-stream-wind frequency histogram.

    ``freq`` is ``(n_direction, n_speed)`` and sums to 1; ``wd_edges`` and
    ``ws_edges`` are the bin edges (each one longer than the matching ``freq``
    axis). Bin representatives are the edge midpoints (:meth:`centers`).
    """

    freq: NDArray[np.float64]
    wd_edges: NDArray[np.float64]
    ws_edges: NDArray[np.float64]

    def centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-axis bin midpoints ``(wd_values, ws_values)`` used as eval winds."""
        wd = (self.wd_edges[:-1] + self.wd_edges[1:]) / 2.0
        ws = (self.ws_edges[:-1] + self.ws_edges[1:]) / 2.0
        return wd, ws

    def bins(self) -> Iterator[tuple[int, int, float, float]]:
        """Yield ``(i, j, wind_direction, wind_speed)`` for every histogram cell."""
        wd_values, ws_values = self.centers()
        for i, wd in enumerate(wd_values):
            for j, ws in enumerate(ws_values):
                yield i, j, float(wd), float(ws)

    def weighted(self, per_bin: NDArray[np.float64]) -> float:
        """Frequency-weighted sum of a per-bin quantity (must match ``freq`` shape)."""
        if per_bin.shape != self.freq.shape:
            raise ValueError(
                f"per_bin shape {per_bin.shape} != rose shape {self.freq.shape}"
            )
        return float(np.sum(self.freq * per_bin))


def prepare_wind_rose(
    wd: NDArray[np.float64],
    ws: NDArray[np.float64],
    num_bins: int = 5,
    direction_offset: float = SMARTEOLE_DIRECTION_OFFSET,
) -> WindRose:
    """Build a :class:`WindRose` from raw wind-direction/speed samples.

    Directions are wrapped to ``[0, 360)``, rotated by ``direction_offset``, and
    wrapped again; the pair is then binned into ``num_bins x num_bins`` cells with
    :func:`numpy.histogram2d`. Frequencies are the normalised cell counts.
    """
    wd_shifted = (np.asarray(wd, dtype=np.float64) % 360.0 + direction_offset) % 360.0
    ws_arr = np.asarray(ws, dtype=np.float64)
    counts, wd_edges, ws_edges = np.histogram2d(wd_shifted, ws_arr, bins=num_bins)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("wind rose histogram is empty (no samples binned)")
    return WindRose(counts / total, wd_edges, ws_edges)


class WindRoseEvalConfig(Config):
    """A precomputed wind rose the trainer sweeps at eval time.

    Carries the histogram inline (frequencies + edges) so a variant fully
    declares its eval wind distribution in config, with no runtime file IO. Build
    one from a :class:`WindRose` via :meth:`from_rose`.
    """

    freq: list[list[float]] = Field(min_length=1)
    wd_edges: list[float] = Field(min_length=2)
    ws_edges: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def _shapes_and_normalisation(self) -> WindRoseEvalConfig:
        rows = len(self.freq)
        cols = len(self.freq[0])
        if any(len(row) != cols for row in self.freq):
            raise ValueError("freq rows must all have the same length")
        if len(self.wd_edges) != rows + 1:
            raise ValueError(
                f"wd_edges must have {rows + 1} entries for {rows} direction bins, "
                f"got {len(self.wd_edges)}"
            )
        if len(self.ws_edges) != cols + 1:
            raise ValueError(
                f"ws_edges must have {cols + 1} entries for {cols} speed bins, "
                f"got {len(self.ws_edges)}"
            )
        total = sum(sum(row) for row in self.freq)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"freq must sum to 1, got {total}")
        return self

    def to_rose(self) -> WindRose:
        return WindRose(
            np.asarray(self.freq, dtype=np.float64),
            np.asarray(self.wd_edges, dtype=np.float64),
            np.asarray(self.ws_edges, dtype=np.float64),
        )

    @classmethod
    def from_rose(cls, rose: WindRose) -> WindRoseEvalConfig:
        return cls(
            freq=rose.freq.tolist(),
            wd_edges=rose.wd_edges.tolist(),
            ws_edges=rose.ws_edges.tolist(),
        )
