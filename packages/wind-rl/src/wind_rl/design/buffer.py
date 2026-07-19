"""File-backed, lock-protected layout buffer shared across processes.

A :class:`LayoutProducer` (main process) pushes generated layout batches; one
:class:`LayoutConsumer` per env worker pops a single layout at each reset. The
buffer lives under a run-scoped directory derived from :class:`WindRlSettings`,
guarded by a shared :class:`multiprocessing.Lock` so producer and consumers can
race safely across forked processes.
"""

from __future__ import annotations

import multiprocessing
import pickle
from multiprocessing.synchronize import Lock
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wind_rl.experiment.settings import WindRlSettings

_BUFFER_FILENAME = "layout_buffer.pkl"


def run_buffer_dir(run_id: str, settings: WindRlSettings | None = None) -> Path:
    """Run-scoped layout-buffer directory under the configured ``wdir``."""
    settings = settings or WindRlSettings()
    return settings.resolved_wdir / "layouts" / run_id


class _LayoutBuffer:
    def __init__(self, directory: Path, lock: Lock) -> None:
        self._lock = lock
        self._path = directory / _BUFFER_FILENAME

    def _read(self) -> list[NDArray[np.float64]]:
        with self._path.open("rb") as f:
            buffer: list[NDArray[np.float64]] = pickle.load(f)
        return buffer

    def _write(self, buffer: list[NDArray[np.float64]]) -> None:
        with self._path.open("wb") as f:
            pickle.dump(buffer, f)


class LayoutProducer(_LayoutBuffer):
    def __init__(self, directory: Path, lock: Lock) -> None:
        super().__init__(directory, lock)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._write([])

    def push(self, batch: NDArray[np.float64]) -> None:
        """Append the ``(B, N, 2)`` layouts to the buffer."""
        layouts = [np.asarray(layout, dtype=np.float64) for layout in batch]
        with self._lock:
            buffer = self._read()
            buffer.extend(layouts)
            self._write(buffer)


class LayoutConsumer(_LayoutBuffer):
    def pop(self) -> NDArray[np.float64] | None:
        """Remove and return one ``(N, 2)`` layout, or ``None`` if empty."""
        with self._lock:
            buffer = self._read()
            if not buffer:
                return None
            layout = buffer.pop()
            self._write(buffer)
        return layout


def create_layout_buffer(directory: Path) -> tuple[LayoutProducer, LayoutConsumer]:
    """A producer/consumer pair over ``directory`` sharing one lock."""
    lock = multiprocessing.Lock()
    return LayoutProducer(directory, lock), LayoutConsumer(directory, lock)
