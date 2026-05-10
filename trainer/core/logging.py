"""
trainer/core/logging.py — Centralized logging configuration.

Provides a consistent format across leader, worker, and tools.
Call setup() once at process start; use get() everywhere else.
"""
from __future__ import annotations
import logging
import sys

_FMT    = "%(asctime)s [%(name)s] %(levelname)s  %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def setup(role: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure the root logger and return a named logger for *role*.
    Idempotent — safe to call from multiple modules.
    """
    global _configured
    root = logging.getLogger()
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        # Clear any handlers added by basicConfig before we got here
        root.handlers.clear()
        root.addHandler(handler)
        _configured = True
    root.setLevel(level)
    return logging.getLogger(role)


def get(role: str) -> logging.Logger:
    """Return a named logger without reconfiguring the root."""
    return logging.getLogger(role)
