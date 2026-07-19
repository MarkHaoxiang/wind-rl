import wind_rl


def test_version() -> None:
    assert wind_rl.__version__ == "0.1.0"


def test_main_is_callable() -> None:
    assert callable(wind_rl.main)
