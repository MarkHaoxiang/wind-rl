"""Local replay server: stdlib HTTP, on-demand fields, one self-contained page.

``python -m windrl_engine.viz path/to/episode.npz [--port N]`` loads a recorded
episode, serves its stats as JSON and its per-frame wake fields as raw float32
bytes, and prints a URL for the bundled canvas viewer.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import cast, override
from urllib.parse import parse_qs, urlparse

import numpy as np

from windrl_engine.viz.field import EpisodeFields
from windrl_engine.viz.record import EpisodeRecord, load_record


def meta_payload(record: EpisodeRecord, fields: EpisodeFields) -> dict[str, object]:
    ny, nx = fields.shape
    power_max = float(record.power.max()) if record.power.size else 1.0
    return {
        "n_frames": int(record.yaw.shape[0]),
        "n_turbines": int(record.yaw.shape[1]),
        "layout_x": record.layout_x.tolist(),
        "layout_y": record.layout_y.tolist(),
        "extent": list(fields.extent),
        "field_shape": [ny, nx],
        "field_vmin": 0.0,
        "field_vmax": float(record.wind_speed.max()) if record.wind_speed.size else 1.0,
        "rotor_diameter": record.rotor_diameter,
        "seconds_per_step": record.seconds_per_step,
        "power_max": power_max if power_max > 0.0 else 1.0,
        "frames": {
            "yaw": record.yaw.tolist(),
            "power": record.power.tolist(),
            "reward": record.reward.tolist(),
            "wind_speed": record.wind_speed.tolist(),
            "wind_direction": record.wind_direction.tolist(),
            "truncated": record.truncated.astype(bool).tolist(),
            "step_count": record.step_count.tolist(),
        },
    }


def field_bytes(fields: EpisodeFields, frame: int) -> bytes:
    clamped = max(0, min(frame, fields.n_frames - 1))
    return np.ascontiguousarray(fields.field_at(clamped), dtype="<f4").tobytes()


def _app_html() -> str:
    return (files("windrl_engine.viz") / "app.html").read_text(encoding="utf-8")


class ReplayServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], record: EpisodeRecord, fields: EpisodeFields
    ) -> None:
        super().__init__(address, ReplayHandler)
        self.record = record
        self.fields = fields
        self.html = _app_html().encode("utf-8")


class ReplayHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        server = cast(ReplayServer, self.server)
        route = urlparse(self.path)
        if route.path == "/":
            self._send(200, "text/html; charset=utf-8", server.html)
        elif route.path == "/api/meta":
            body = json.dumps(meta_payload(server.record, server.fields)).encode(
                "utf-8"
            )
            self._send(200, "application/json", body)
        elif route.path == "/api/field":
            frame = _query_frame(route.query)
            self._send(
                200, "application/octet-stream", field_bytes(server.fields, frame)
            )
        else:
            self._send(404, "text/plain", b"not found")

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _query_frame(query: str) -> int:
    values = parse_qs(query).get("frame", ["0"])
    try:
        return int(values[0])
    except ValueError:
        return 0


def serve(
    record: EpisodeRecord, *, host: str = "127.0.0.1", port: int = 8000
) -> ReplayServer:
    fields = EpisodeFields(record)
    return ReplayServer((host, port), record, fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded wind-farm episode.")
    parser.add_argument("episode", help="path to an EpisodeRecord .npz")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    record = load_record(args.episode)
    server = serve(record, host=args.host, port=args.port)
    host, port = server.server_address[0], server.server_address[1]
    host_str = host.decode() if isinstance(host, bytes) else host
    print(f"episode replay: http://{host_str}:{port}/  ({record.yaw.shape[0]} frames)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
