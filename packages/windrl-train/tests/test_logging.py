import sys

import pytest

from windrl_train.logging import ConsoleLogger, Logger, NullLogger, WandbLogger


def test_null_logger_satisfies_logger_protocol() -> None:
    assert isinstance(NullLogger(), Logger)


def test_console_logger_satisfies_logger_protocol() -> None:
    assert isinstance(ConsoleLogger(), Logger)


def test_wandb_logger_class_satisfies_logger_protocol() -> None:
    assert issubclass(WandbLogger, Logger)


def test_null_logger_records_log_stat_calls() -> None:
    logger = NullLogger()
    logger.log_stat("reward", 1.5, step=10, event="train")
    logger.log_stat("loss", 0.2, step=20, event="eval")
    assert logger.stats == [
        ("reward", 1.5, 10, "train"),
        ("loss", 0.2, 20, "eval"),
    ]


def test_null_logger_log_config_and_stop_are_no_ops() -> None:
    logger = NullLogger()
    logger.log_config({"lr": 0.001})
    logger.stop()
    assert logger.stats == []


def test_console_logger_prints_step_event_key_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = ConsoleLogger()
    logger.log_stat("reward", 1.5, step=10, event="train")
    out = capsys.readouterr().out
    assert "step=10" in out
    assert "train/reward=1.5" in out


def test_importing_logging_package_does_not_initialize_wandb() -> None:
    assert "wandb" not in sys.modules
