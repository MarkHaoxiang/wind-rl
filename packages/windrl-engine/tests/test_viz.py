"""viz/ (episode recording, on-demand fields, replay server), CPU + tiny farm."""

import json
import threading
from urllib.request import urlopen

import jax
import numpy as np

from windrl_engine.env.config import WindFarmEnvConfig
from windrl_engine.env.env import BatchedWindFarmEnv
from windrl_engine.viz.field import EpisodeFields
from windrl_engine.viz.record import (
    EpisodeRecord,
    load_record,
    record_episode,
    save_record,
    sweeping_actor,
)
from windrl_engine.viz.server import field_bytes, meta_payload, serve


def _record(n_steps: int = 6) -> EpisodeRecord:
    env = BatchedWindFarmEnv(WindFarmEnvConfig(layout="turb3_row1", n_envs=2))
    return record_episode(
        env, jax.random.key(0), n_steps, sweeping_actor(env.config.yaw_step)
    )


def test_record_has_frame_per_step_plus_reset_with_matching_shapes() -> None:
    record = _record(n_steps=6)
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


def test_sweeping_actor_visibly_moves_yaw_off_zero() -> None:
    record = _record(n_steps=8)
    # frame 0 is the zero-yaw reset; the outer turbines must ramp away from it.
    assert float(np.abs(record.yaw[-1]).max()) > 10.0


def test_save_load_round_trips_every_field_exactly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    record = _record()
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


def test_field_is_finite_and_bounded_by_freestream() -> None:
    record = _record()
    fields = EpisodeFields(record, resolution=(48, 48))
    field = fields.field_at(2)
    assert field.shape == (48, 48)
    assert np.all(np.isfinite(field))
    # a wake only removes momentum, so hub-height u stays within [0, freestream].
    freestream = float(record.wind_speed.max())
    assert field.min() >= 0.0
    assert field.max() <= freestream + 1e-3


def test_field_cache_returns_the_identical_array() -> None:
    fields = EpisodeFields(_record(), resolution=(32, 32))
    assert fields.field_at(1) is fields.field_at(1)


def test_meta_payload_omits_fields_unused_by_the_viewer() -> None:
    record = _record()
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


def test_field_bytes_decode_to_the_field_shape() -> None:
    record = _record()
    fields = EpisodeFields(record, resolution=(40, 50))
    ny, nx = fields.shape
    assert (ny, nx) == (50, 40)
    raw = field_bytes(fields, 3)
    decoded = np.frombuffer(raw, dtype="<f4")
    assert decoded.size == ny * nx
    assert np.array_equal(decoded.reshape(ny, nx), fields.field_at(3))


def test_live_server_serves_page_meta_and_field_frames() -> None:
    record = _record()
    server = serve(record, port=0)
    server.fields = EpisodeFields(record, resolution=(32, 32))
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
