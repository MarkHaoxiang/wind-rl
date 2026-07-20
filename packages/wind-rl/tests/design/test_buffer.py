from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np
import pytest

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


def test_interrupted_write_leaves_previous_buffer_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer, consumer = create_layout_buffer(tmp_path)
    good = np.zeros((1, 3, 2), dtype=np.float64)
    producer.push(good)

    def exploding_dump(obj: object, file: object, *args: object, **kw: object) -> None:
        file.write(b"partial garbage")  # type: ignore[attr-defined]
        raise RuntimeError("crash mid-write")

    monkeypatch.setattr(pickle, "dump", exploding_dump)
    with pytest.raises(RuntimeError):
        producer.push(np.ones((1, 3, 2), dtype=np.float64))
    monkeypatch.undo()

    # os.replace never ran, so the target still holds the pre-crash buffer, and
    # the failed write left no temp file behind for a reader to trip over.
    assert not list(tmp_path.glob("*.tmp*"))
    survivor = consumer.pop()
    assert survivor is not None
    np.testing.assert_array_equal(survivor, good[0])
    # The failed push appended nothing.
    assert consumer.pop() is None


def test_no_temp_file_remains_after_successful_push(tmp_path: Path) -> None:
    producer, _ = create_layout_buffer(tmp_path)
    producer.push(np.zeros((3, 2, 2), dtype=np.float64))
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.lock"))


def test_concurrent_push_pop_loses_no_layouts(tmp_path: Path) -> None:
    producer, consumer = create_layout_buffer(tmp_path)
    n_threads, per_thread = 8, 25

    def worker(tid: int) -> None:
        for i in range(per_thread):
            value = float(tid * per_thread + i)
            producer.push(np.full((1, 2, 2), value, dtype=np.float64))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The lock must serialize read-modify-write; without it concurrent pushes
    # would clobber each other and drop entries.
    popped: list[float] = []
    while (layout := consumer.pop()) is not None:
        popped.append(float(layout[0, 0]))
    assert sorted(popped) == [float(v) for v in range(n_threads * per_thread)]


def test_written_buffer_is_a_complete_pickle(tmp_path: Path) -> None:
    producer, _ = create_layout_buffer(tmp_path)
    producer.push(np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2))
    with (tmp_path / "layout_buffer.pkl").open("rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 4
    assert all(layout.shape == (2, 2) for layout in loaded)
