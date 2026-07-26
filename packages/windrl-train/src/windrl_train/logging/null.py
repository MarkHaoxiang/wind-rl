from typing import Any


class NullLogger:
    """No-op logger; records every ``log_stat`` call in ``.stats`` for tests."""

    def __init__(self) -> None:
        self.stats: list[tuple[str, float, int, str]] = []

    def log_stat(self, key: str, value: float, step: int, event: str) -> None:
        self.stats.append((key, value, step, event))

    def log_config(self, config: dict[str, Any]) -> None:
        pass

    def stop(self) -> None:
        pass
