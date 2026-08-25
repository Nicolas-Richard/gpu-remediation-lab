"""Host-side HTTP control for one in-cluster DCGM metrics simulator pod."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import HarnessError


class SimulatorClient:
    def __init__(self, base_url: str, request_timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout

    def health(self) -> None:
        self._request("GET", "/healthz")

    def set_xid(self, xid: int) -> None:
        self._request("PUT", "/state", {"xid": xid})

    def clear(self) -> None:
        self._request("DELETE", "/state")

    def metrics(self) -> str:
        return self._request("GET", "/metrics").decode()

    def _request(
        self,
        method: str,
        path: str,
        document: Optional[dict[str, object]] = None,
    ) -> bytes:
        body = None
        headers: dict[str, str] = {}
        if document is not None:
            body = json.dumps(document).encode()
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                return response.read()
        except HTTPError as error:
            detail = error.read().decode(errors="replace").strip()
            raise HarnessError(
                f"simulator {method} {path} returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise HarnessError(
                f"simulator {method} {path} failed: {error}"
            ) from error


class SimulatorPortForward:
    """Own a temporary localhost tunnel to one specific simulator pod."""

    def __init__(
        self,
        context: str,
        namespace: str,
        pod_name: str,
        *,
        remote_port: int = 9400,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.pod_name = pod_name
        self.remote_port = remote_port
        self.local_port: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 15.0) -> SimulatorClient:
        if self.process is not None:
            raise HarnessError("simulator port-forward is already running")
        self.local_port = self._available_local_port()
        command = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            "port-forward",
            f"pod/{self.pod_name}",
            f"{self.local_port}:{self.remote_port}",
            "--address=127.0.0.1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = SimulatorClient(f"http://127.0.0.1:{self.local_port}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.process.communicate()[0].strip()
                self.process = None
                raise HarnessError(
                    f"simulator port-forward exited before it was ready: {output}"
                )
            try:
                client.health()
                return client
            except HarnessError:
                time.sleep(0.1)
        self.close()
        raise HarnessError(
            f"timed out waiting for port-forward to simulator pod {self.pod_name}"
        )

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()

    @staticmethod
    def _available_local_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])
