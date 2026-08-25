#!/usr/bin/env python3
"""Small, controllable DCGM Prometheus endpoint for local failure tests."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlsplit


METRIC_NAME = "DCGM_FI_DEV_XID_ERRORS"
MAX_REQUEST_BYTES = 1024


class XIDState:
    """Thread-safe storage for the simulator's latest XID observation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._xid: Optional[int] = None

    def get(self) -> Optional[int]:
        with self._lock:
            return self._xid

    def set(self, xid: int) -> None:
        with self._lock:
            self._xid = xid

    def clear(self) -> None:
        with self._lock:
            self._xid = None


def parse_state_payload(payload: bytes) -> int:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"xid"}:
        raise ValueError('body must be exactly {"xid": <integer>}')
    xid = document["xid"]
    if isinstance(xid, bool) or not isinstance(xid, int):
        raise ValueError("xid must be an integer")
    if xid < 0 or xid > 999:
        raise ValueError("xid must be between 0 and 999")
    return xid


def prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_metrics(xid: Optional[int], node_name: str) -> str:
    if xid is None:
        return "# DCGM XID observation absent\n"
    node = prometheus_escape(node_name)
    uuid = prometheus_escape(f"GPU-SIM-{node_name}-0")
    return (
        f"# HELP {METRIC_NAME} Last observed NVIDIA XID error.\n"
        f"# TYPE {METRIC_NAME} gauge\n"
        f'{METRIC_NAME}{{gpu="0",UUID="{uuid}",device="nvidia0",'
        f'Hostname="{node}"}} {xid}\n'
    )


class SimulatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: XIDState, node_name: str):
        super().__init__(address, SimulatorRequestHandler)
        self.state = state
        self.node_name = node_name


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    server: SimulatorHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._write(200, b"ok\n", "text/plain; charset=utf-8")
            return
        if path == "/metrics":
            metrics = render_metrics(self.server.state.get(), self.server.node_name)
            self._write(
                200,
                metrics.encode(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
            return
        if path == "/state":
            self._write_json(200, {"xid": self.server.state.get()})
            return
        self._write_json(404, {"error": "not found"})

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if urlsplit(self.path).path != "/state":
            self._write_json(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(400, {"error": "invalid Content-Length"})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._write_json(
                400,
                {"error": f"body must contain 1-{MAX_REQUEST_BYTES} bytes"},
            )
            return
        try:
            xid = parse_state_payload(self.rfile.read(content_length))
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return
        self.server.state.set(xid)
        self._write_json(200, {"xid": xid})

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if urlsplit(self.path).path != "/state":
            self._write_json(404, {"error": "not found"})
            return
        self.server.state.clear()
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _write_json(self, status: int, document: dict[str, object]) -> None:
        self._write(
            status,
            (json.dumps(document, sort_keys=True) + "\n").encode(),
            "application/json; charset=utf-8",
        )

    def _write(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message: str, *arguments: object) -> None:
        print(
            f'{self.address_string()} - [{self.log_date_time_string()}] '
            f'{message % arguments}',
            flush=True,
        )


def main() -> None:
    node_name = os.environ.get("NODE_NAME", "unknown-node")
    port = int(os.environ.get("PORT", "9400"))
    server = SimulatorHTTPServer(("0.0.0.0", port), XIDState(), node_name)
    print(
        f"DCGM metrics simulator listening on :{port} for node {node_name}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
