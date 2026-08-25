"""Fault-recovery tests for physical GPUs on AWS."""


class DCGMInjectionError(RuntimeError):
    """The AWS DCGM injection scenario could not prove a required transition."""
