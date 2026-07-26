from typing import Any, Protocol, runtime_checkable

from windrl_train.logging.console import ConsoleLogger
from windrl_train.logging.null import NullLogger
from windrl_train.logging.wandb import WandbLogger


@runtime_checkable
class Logger(Protocol):
    """A stat sink experiments log to, decoupling trainer code from wandb."""

    def log_stat(self, key: str, value: float, step: int, event: str) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...
    def stop(self) -> None: ...


__all__ = ["ConsoleLogger", "Logger", "NullLogger", "WandbLogger"]
