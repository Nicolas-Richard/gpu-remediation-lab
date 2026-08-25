"""Structured scenario events with a compact compose-style human renderer."""

from __future__ import annotations

import json
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TextIO


def short_uid(value: str) -> str:
    """Keep pod identity readable while preserving enough entropy for a run."""
    return value[:8]


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


class NodeAliases:
    """Assign stable, run-local names in first-observation order."""

    def __init__(self, prefix: str = "gpu") -> None:
        self.prefix = prefix
        self._aliases: dict[str, str] = {}

    def get(self, node: str) -> str:
        if node not in self._aliases:
            index = len(self._aliases)
            suffix = (
                string.ascii_lowercase[index]
                if index < len(string.ascii_lowercase)
                else str(index + 1)
            )
            self._aliases[node] = f"{self.prefix}-{suffix}"
        return self._aliases[node]

    def mapping(self) -> dict[str, str]:
        return {alias: node for node, alias in self._aliases.items()}


class ScenarioReporter:
    """Render one event model as readable text and optional JSON Lines."""

    COMPONENT_WIDTH = 28

    def __init__(
        self,
        run_id: str,
        *,
        jsonl_path: Optional[Path] = None,
        output: Optional[TextIO] = None,
        verbose: bool = False,
        wall_clock: Optional[Callable[[], datetime]] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        self.run_id = run_id
        self.jsonl_path = jsonl_path
        self.output = output or sys.stdout
        self.verbose = verbose
        self.wall_clock = wall_clock or (lambda: datetime.now().astimezone())
        self.monotonic = monotonic or time.monotonic
        self.started_at = self.monotonic()
        self.sequence = 0
        self.current_phase = "startup"
        self.nodes = NodeAliases()
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("")

    def node(self, hostname: str) -> str:
        return self.nodes.get(hostname)

    def phase(self, name: str) -> None:
        self.current_phase = name
        self.emit(
            "harness",
            "PHASE",
            message=f"Entered {name} phase",
            visible=False,
            name=name,
        )

    def emit(
        self,
        component: str,
        event: str,
        *,
        message: Optional[str] = None,
        visible: bool = True,
        **fields: object,
    ) -> None:
        self.sequence += 1
        now = self.wall_clock()
        elapsed = self.monotonic() - self.started_at
        if visible or self.verbose:
            details = " ".join(
                f"{key}={self._human_value(value)}" for key, value in fields.items()
            )
            rendered = message or f"{event} {details}".rstrip()
            component_prefix = f"[{component}]".ljust(self.COMPONENT_WIDTH + 2)
            prefix = (
                f"{now.strftime('%H:%M:%S')} +{elapsed:05.1f}s "
                f"{component_prefix}"
            )
            print(f"{prefix} {rendered}".rstrip(), file=self.output, flush=True)

        if self.jsonl_path is not None:
            payload = {
                "timestamp": now.astimezone().isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "sequence": self.sequence,
                "run_id": self.run_id,
                "phase": self.current_phase,
                "component": component,
                "event": event,
                **fields,
            }
            if message is not None:
                payload["message"] = message
            with self.jsonl_path.open("a") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def raw(self, component: str, text: str) -> None:
        for line in text.rstrip().splitlines():
            self.emit(component, "LOG", line=line)

    @staticmethod
    def _human_value(value: object) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, separators=(",", ":"), sort_keys=True)
        rendered = str(value)
        if any(char.isspace() for char in rendered):
            return json.dumps(rendered)
        return rendered
