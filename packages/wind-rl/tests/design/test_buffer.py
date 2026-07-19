from __future__ import annotations

from pathlib import Path

import numpy as np

from wind_rl.design import create_layout_buffer


def test_buffer_round_trip(tmp_path: Path) -> None:
    producer, consumer = create_layout_buffer(tmp_path)
    batch = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)

    producer.push(batch)
    # ``pop`` is LIFO, so it returns the pushed layouts in reverse order.
    last, first = consumer.pop(), consumer.pop()
    assert first is not None and last is not None

    np.testing.assert_array_equal(np.stack([first, last]), batch)


def test_pop_on_empty_returns_none(tmp_path: Path) -> None:
    _, consumer = create_layout_buffer(tmp_path)
    assert consumer.pop() is None
