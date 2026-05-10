"""
trainer/core/errors.py — Custom exceptions + gRPC error translator.

Instead of a 100-line gRPC stack trace, users see:
    LeaderConnectionError: UNAVAILABLE: failed to connect to all addresses
"""
from __future__ import annotations


class DistributedTrainerError(Exception):
    """Base for all distributed-trainer errors."""


class LeaderConnectionError(DistributedTrainerError):
    """Cannot connect to or register with the leader."""


class RegistrationRejectedError(DistributedTrainerError):
    """Leader explicitly rejected the worker registration."""


class GradientSyncError(DistributedTrainerError):
    """Gradient exchange with the leader failed."""


class AssignmentError(DistributedTrainerError):
    """Failed to receive a training assignment from the leader."""


class BootstrapError(DistributedTrainerError):
    """venv / dependency setup failed."""


def fmt_grpc_error(exc: BaseException) -> str:
    """One-line human-readable description of any gRPC or timeout error."""
    try:
        import grpc.aio as _aio
        if isinstance(exc, _aio.AioRpcError):
            return f"{exc.code().name}: {exc.details()}"
    except Exception:
        pass
    try:
        import grpc as _grpc
        if isinstance(exc, _grpc.RpcError):
            return f"{exc.code().name}: {exc.details()}"
    except Exception:
        pass
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timed out waiting for gRPC channel to become ready"
    return f"{type(exc).__name__}: {exc}"


# Lazy import so this module stays importable before asyncio is set up
try:
    import asyncio
except ImportError:
    pass
