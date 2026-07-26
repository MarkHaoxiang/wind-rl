from typing import Any


class ConsoleLogger:
    """Prints every stat to stdout instead of a tracking backend."""

    def log_stat(self, key: str, value: float, step: int, event: str) -> None:
        print(f"step={step} {event}/{key}={value}")

    def log_config(self, config: dict[str, Any]) -> None:
        print(f"config={config}")

    def stop(self) -> None:
        pass
