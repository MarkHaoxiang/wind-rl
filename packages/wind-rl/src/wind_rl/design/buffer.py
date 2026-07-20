"""File-backed, lock-protected layout buffer shared across processes.

A :class:`LayoutProducer` (main process) pushes generated layout batches; one
:class:`LayoutConsumer` per env worker pops a single layout at each reset. The
buffer lives under a run-scoped directory derived from :class:`WindRlSettings`.

Cross-process safety is start-method agnostic (works under both ``fork`` and
``spawn``): the mutual exclusion is an ``O_CREAT | O_EXCL`` lock file rather than
a :class:`multiprocessing.Lock` (which is only valid for fork-children that
inherit the object). Producer and consumer hold only a directory path, so they
pickle cleanly into a spawned worker's env factory. Writes are atomic (temp file
+ :func:`os.replace`) so a crash mid-write can never leave a truncated pickle
that a later reader would choke on.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from wind_rl.experiment.settings import WindRlSettings

_BUFFER_FILENAME = "layout_buffer.pkl"
_LOCK_FILENAME = "layout_buffer.lock"
_LOCK_POLL_S = 0.001


def run_buffer_dir(run_id: str, settings: WindRlSettings | None = None) -> Path:
    """Run-scoped layout-buffer directory under the configured ``wdir``."""
    settings = settings or WindRlSettings()
    return settings.resolved_wdir / "layouts" / run_id


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> _FileLock:
        while True:
            try:
                self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                time.sleep(_LOCK_POLL_S)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._path.unlink(missing_ok=True)


class _LayoutBuffer:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._path = directory / _BUFFER_FILENAME
        self._lock = _FileLock(directory / _LOCK_FILENAME)

    def _read(self) -> list[NDArray[np.float64]]:
        with self._path.open("rb") as f:
            buffer: list[NDArray[np.float64]] = pickle.load(f)
        return buffer

    def _write(self, buffer: list[NDArray[np.float64]]) -> None:
        # Temp file + os.replace: the target is swapped atomically, so a crash
        # (or an interrupted pickle.dump) leaves the previous complete buffer in
        # place -- a reader never observes a partial file.
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=f"{_BUFFER_FILENAME}.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(buffer, f)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class LayoutProducer(_LayoutBuffer):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
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
    """A producer/consumer pair over ``directory`` (shared via an on-disk lock)."""
    return LayoutProducer(directory), LayoutConsumer(directory)
