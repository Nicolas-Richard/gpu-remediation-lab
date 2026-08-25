"""Bounded polling shared by local and AWS recovery harnesses."""

from __future__ import annotations

import time
from typing import Callable, Optional, Type, TypeVar

from . import RecoveryError


T = TypeVar("T")


def wait_for(
    operation: Callable[[], Optional[T]],
    description: str,
    *,
    timeout: float,
    interval: float = 2.0,
    error_type: Type[RuntimeError] = RecoveryError,
) -> T:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = operation()
        if result is not None:
            return result
        time.sleep(interval)
    raise error_type(f"timed out after {timeout:g}s waiting for {description}")
