"""viz/ (episode recording, on-demand fields, replay server), CPU + tiny farm."""

import json
import threading
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from windrl_engine.env.actions import Fidelity
from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import BatchedWindFarmEnv
from windrl_engine.farm.wind import WindCondition
from windrl_engine.physics.power import turbine_powers
from windrl_engine.physics.solver import solve_farm
from windrl_engine.viz.field import CACHED_FRAMES, EpisodeFields
from windrl_engine.viz.record import (
    EpisodeRecord,
    load_record,
    record_episode,
    save_record,
    sweeping_actor,
)
from windrl_engine.viz.server import field_bytes, meta_payload, serve


def _record(n_steps: int = 6, fidelity: Fidelity = "floris") -> EpisodeRecord:
    env = BatchedWindFarmEnv(
        WindFarmEnvConfig(layout="turb3_row1", n_envs=2, fidelity=fidelity)
    )
    return record_episode(
        env, jax.random.key(0), n_steps, sweeping_actor(env.config.yaw_step)
    )


# Recording an episode dominates this module's runtime (env construction plus a
# jitted rollout); EpisodeRecord is an immutable NamedTuple of numpy arrays that
# no test writes to, so one 7-frame episode serves every test that only reads it.
@pytest.fixture(scope="module")
def record() -> EpisodeRecord:
    return _record()


@pytest.fixture(scope="module")
def corrected_record() -> EpisodeRecord:
    return _record(fidelity="corrected")


def test_record_has_frame_per_step_plus_reset_with_matching_shapes(
    record: EpisodeRecord,
) -> None:
    frames, turbines = 7, 3
    assert record.yaw.shape == (frames, turbines)
    assert record.power.shape == (frames, turbines)
    assert record.action.shape == (frames, turbines)
    assert record.reward.shape == (frames,)
    assert record.wind_speed.shape == (frames,)
    assert record.step_count.shape == (frames,)
    assert record.yaw.dtype == np.float32
    assert record.truncated.dtype == np.bool_
    assert record.step_count.dtype == np.int32
    assert np.all(record.power >= 0.0)


def test_sweeping_actor_visibly_moves_yaw_off_zero(record: EpisodeRecord) -> None:
    # frame 0 is the zero-yaw reset; the outer turbines must ramp away from it
    # (duty-limited to a move every other step, so 6 steps reach +/-15 deg).
    assert float(np.abs(record.yaw[-1]).max()) > 10.0


def test_save_load_round_trips_every_field_exactly(
    record: EpisodeRecord, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "episode.npz"
    save_record(record, path)
    loaded = load_record(path)
    for name in record._fields:
        original = getattr(record, name)
        restored = getattr(loaded, name)
        if isinstance(original, np.ndarray):
            assert np.array_equal(original, restored)
            assert original.dtype == restored.dtype
        else:
            assert original == restored


def test_recorded_power_is_the_fidelity_the_env_was_rewarded_under(
    corrected_record: EpisodeRecord,
) -> None:
    # "floris" and "corrected" differ by ~6 kW/turbine here, so recomputing the
    # frame powers at the wrong fidelity is visible far above float32 noise.
    record = corrected_record
    env = BatchedWindFarmEnv(
        WindFarmEnvConfig(layout="turb3_row1", n_envs=2, fidelity="corrected")
    )
    last = record.yaw.shape[0] - 1
    wind = WindCondition(
        speed=jnp.asarray(record.wind_speed[last]),
        direction=jnp.asarray(record.wind_direction[last]),
    )
    yaw = jnp.asarray(record.yaw[last])

    def powers_at(fidelity: Fidelity) -> np.ndarray:  # type: ignore[type-arg]
        solution = solve_farm(
            env.layout, wind, yaw, fidelity=fidelity, turbine=env.turbine
        )
        return np.asarray(turbine_powers(solution.u, yaw, turbine=env.turbine))

    assert record.fidelity == "corrected"
    np.testing.assert_allclose(record.power[last], powers_at("corrected"), rtol=1e-6)
    assert np.abs(record.power[last] - powers_at("floris")).max() > 1e3


def test_a_record_saved_without_a_fidelity_field_loads_as_the_reference_model(
    record: EpisodeRecord,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "legacy.npz"
    fields = {k: v for k, v in record._asdict().items() if k != "fidelity"}
    np.savez(path, **fields)
    assert load_record(path).fidelity == "floris"


def test_field_is_finite_and_bounded_by_freestream(record: EpisodeRecord) -> None:
    fields = EpisodeFields(record, resolution=(48, 48))
    field = fields.field_at(2)
    assert field.shape == (48, 48)
    assert np.all(np.isfinite(field))
    # a wake only removes momentum, so hub-height u stays within [0, freestream].
    freestream = float(record.wind_speed.max())
    assert field.min() >= 0.0
    assert field.max() <= freestream + 1e-3


def test_field_cache_returns_the_identical_array(record: EpisodeRecord) -> None:
    fields = EpisodeFields(record, resolution=(32, 32))
    assert fields.field_at(1) is fields.field_at(1)


def test_field_cache_evicts_the_least_recent_frame_once_full() -> None:
    # A long scrub must not pin every frame it passed: at ~230 kB each, an
    # unbounded cache retains the whole episode.
    fields = EpisodeFields(_record(n_steps=CACHED_FRAMES + 2), resolution=(8, 8))
    first = fields.field_at(0)
    for frame in range(1, CACHED_FRAMES + 1):
        fields.field_at(frame)
    assert fields.field_at(0) is not first
    assert fields.field_at(CACHED_FRAMES) is fields.field_at(CACHED_FRAMES)


def test_meta_payload_omits_fields_unused_by_the_viewer(record: EpisodeRecord) -> None:
    fields = EpisodeFields(record, resolution=(32, 32))
    payload = meta_payload(record, fields)
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["n_frames"] == record.yaw.shape[0]
    assert round_tripped["n_turbines"] == 3
    assert len(round_tripped["frames"]["power"]) == record.yaw.shape[0]
    assert round_tripped["extent"][0] < round_tripped["extent"][1]
    assert set(round_tripped) == {
        "n_frames",
        "n_turbines",
        "layout_x",
        "layout_y",
        "extent",
        "field_shape",
        "field_vmin",
        "field_vmax",
        "rotor_diameter",
        "seconds_per_step",
        "power_max",
        "frames",
    }
    assert set(round_tripped["frames"]) == {
        "yaw",
        "power",
        "reward",
        "wind_speed",
        "wind_direction",
        "truncated",
        "step_count",
    }


def test_field_bytes_decode_to_the_field_shape(record: EpisodeRecord) -> None:
    fields = EpisodeFields(record, resolution=(40, 50))
    ny, nx = fields.shape
    assert (ny, nx) == (50, 40)
    raw = field_bytes(fields, 3)
    decoded = np.frombuffer(raw, dtype="<f4")
    assert decoded.size == ny * nx
    assert np.array_equal(decoded.reshape(ny, nx), fields.field_at(3))


def test_live_server_serves_page_meta_and_field_frames(record: EpisodeRecord) -> None:
    server = serve(record, port=0, resolution=(32, 32))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        base = f"http://{host}:{port}"
        page = urlopen(f"{base}/").read()
        assert b"<canvas" in page
        meta = json.loads(urlopen(f"{base}/api/meta").read())
        assert meta["n_frames"] == record.yaw.shape[0]
        raw = urlopen(f"{base}/api/field?frame=2").read()
        assert np.frombuffer(raw, dtype="<f4").size == 32 * 32
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
