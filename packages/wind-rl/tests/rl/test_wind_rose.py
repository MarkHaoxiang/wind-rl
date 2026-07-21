from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from wind_rl.rl.wind_rose import (
    WindRose,
    WindRoseEvalConfig,
    prepare_wind_rose,
)


def test_prepare_wind_rose_counts_and_normalisation() -> None:
    wd = np.array([0.0, 0.0, 180.0, 180.0])
    ws = np.array([2.0, 8.0, 2.0, 8.0])
    rose = prepare_wind_rose(wd, ws, num_bins=2, direction_offset=0.0)

    assert rose.freq.shape == (2, 2)
    np.testing.assert_allclose(rose.freq, np.full((2, 2), 0.25))
    assert rose.freq.sum() == pytest.approx(1.0)
    # Edges bracket the data; midpoints are the eval winds.
    wd_c, ws_c = rose.centers()
    np.testing.assert_allclose(wd_c, [45.0, 135.0])
    np.testing.assert_allclose(ws_c, [3.5, 6.5])


def test_prepare_wind_rose_applies_direction_offset() -> None:
    wd = np.array([0.0, 0.0, 180.0, 180.0])
    ws = np.array([2.0, 8.0, 2.0, 8.0])
    rose = prepare_wind_rose(wd, ws, num_bins=2, direction_offset=90.0)

    # +90 deg shifts every direction, so the bin centers move with it.
    wd_c, _ = rose.centers()
    np.testing.assert_allclose(wd_c, [135.0, 225.0])
    np.testing.assert_allclose(rose.freq, np.full((2, 2), 0.25))


def test_prepare_wind_rose_wraps_directions() -> None:
    # 350 + 20 offset wraps to 10; binned together with a genuine 10-deg sample.
    wd = np.array([350.0, -10.0, 10.0, 10.0])
    ws = np.array([5.0, 5.0, 5.0, 5.0])
    rose = prepare_wind_rose(wd, ws, num_bins=1, direction_offset=20.0)
    assert rose.freq.shape == (1, 1)
    assert rose.freq[0, 0] == pytest.approx(1.0)
    assert rose.wd_edges[0] >= 0.0 and rose.wd_edges[-1] < 360.0 + 1e-6


def test_prepare_wind_rose_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_wind_rose(np.array([]), np.array([]), num_bins=2)


def test_weighted_is_frequency_dot_product() -> None:
    freq = np.array([[0.5, 0.25], [0.2, 0.05]])
    rose = WindRose(freq, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    per_bin = np.array([[10.0, 20.0], [30.0, 40.0]])
    # 0.5*10 + 0.25*20 + 0.2*30 + 0.05*40 = 18.0
    assert rose.weighted(per_bin) == pytest.approx(18.0)


def test_weighted_rejects_shape_mismatch() -> None:
    rose = WindRose(
        np.full((2, 2), 0.25), np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    with pytest.raises(ValueError, match="shape"):
        rose.weighted(np.zeros((3, 3)))


def test_bins_enumerates_every_cell_in_order() -> None:
    rose = WindRose(
        np.full((2, 3), 1.0 / 6.0),
        np.array([0.0, 90.0, 180.0]),
        np.array([0.0, 4.0, 8.0, 12.0]),
    )
    idx = [(i, j) for i, j, _wd, _ws in rose.bins()]
    assert idx == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # First cell's representative is the (wd, ws) edge midpoint.
    first = next(rose.bins())
    assert first == (0, 0, 45.0, 2.0)


def test_config_round_trips_through_rose() -> None:
    freq = np.array([[0.6, 0.1], [0.2, 0.1]])
    rose = WindRose(freq, np.array([0.0, 1.0, 2.0]), np.array([2.0, 5.0, 8.0]))
    cfg = WindRoseEvalConfig.from_rose(rose)
    back = cfg.to_rose()
    np.testing.assert_allclose(back.freq, rose.freq)
    np.testing.assert_allclose(back.wd_edges, rose.wd_edges)
    np.testing.assert_allclose(back.ws_edges, rose.ws_edges)


def test_config_rejects_non_normalised_freq() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        WindRoseEvalConfig(
            freq=[[0.5, 0.4]], wd_edges=[0.0, 1.0], ws_edges=[0.0, 1.0, 2.0]
        )


def test_config_rejects_edge_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="wd_edges"):
        WindRoseEvalConfig(
            freq=[[0.5, 0.5]], wd_edges=[0.0, 1.0, 2.0], ws_edges=[0.0, 1.0, 2.0]
        )


def test_config_rejects_ragged_freq() -> None:
    with pytest.raises(ValidationError, match="same length"):
        WindRoseEvalConfig(
            freq=[[0.5, 0.5], [1.0]], wd_edges=[0.0, 1.0, 2.0], ws_edges=[0.0, 1.0, 2.0]
        )
