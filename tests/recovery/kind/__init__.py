"""Fault-recovery test harness for the GPU orchestration demo."""


class HarnessError(RuntimeError):
    """A lifecycle assertion or kubectl command failed."""
