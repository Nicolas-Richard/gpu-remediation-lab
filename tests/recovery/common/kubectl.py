"""Small subprocess-based kubectl client shared by recovery harnesses."""

from __future__ import annotations

import json
import subprocess
from typing import Optional, Sequence, Type

from . import RecoveryError


class Kubectl:
    def __init__(
        self,
        context: str,
        namespace: Optional[str] = None,
        *,
        command_timeout: int = 90,
        error_type: Type[RuntimeError] = RecoveryError,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.command_timeout = command_timeout
        self.error_type = error_type

    def run(
        self,
        arguments: Sequence[str],
        *,
        namespaced: bool = True,
        namespace: Optional[str] = None,
        check: bool = True,
        input_text: Optional[str] = None,
    ) -> str:
        command = ["kubectl", "--context", self.context]
        selected_namespace = namespace if namespace is not None else self.namespace
        if namespaced and selected_namespace:
            command.extend(["--namespace", selected_namespace])
        command.extend(arguments)

        try:
            completed = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise self.error_type(
                f"command timed out after {self.command_timeout}s: {' '.join(command)}"
            ) from error

        if check and completed.returncode != 0:
            output = (completed.stdout + completed.stderr).strip()
            raise self.error_type(
                f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
            )
        return completed.stdout

    def json(
        self,
        arguments: Sequence[str],
        *,
        namespaced: bool = True,
        namespace: Optional[str] = None,
    ) -> dict:
        output = self.run(
            [*arguments, "-o", "json"],
            namespaced=namespaced,
            namespace=namespace,
        )
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise self.error_type(f"kubectl returned invalid JSON: {error}") from error
