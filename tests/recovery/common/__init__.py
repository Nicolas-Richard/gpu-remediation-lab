"""Reusable Kubernetes and recovery-test primitives."""


class RecoveryError(RuntimeError):
    """A Kubernetes command or recovery assertion failed."""
