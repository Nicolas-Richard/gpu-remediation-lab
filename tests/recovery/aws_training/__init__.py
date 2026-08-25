"""Distributed CUDA training recovery validation for AWS EKS."""


class TrainingRecoveryError(RuntimeError):
    """An AWS training-recovery assertion or command failed."""
