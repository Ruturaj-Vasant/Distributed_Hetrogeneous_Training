"""
trainer/leader/worker_state.py — Per-worker bookkeeping.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from proto import trainer_pb2

_DEFAULT_HEARTBEAT_TIMEOUT = 60.0


@dataclass
class WorkerState:
    worker_id:         str
    hostname:          str
    os_name:           str
    score:             float
    cpu_cores:         int
    ram_gb:            float
    accel_summary:     str          # e.g. "CUDA:RTX3060@6GB | MPS:M1@8cores"
    heartbeat_timeout: float = _DEFAULT_HEARTBEAT_TIMEOUT
    registered_at:     float = field(default_factory=time.monotonic)
    last_heartbeat:    float = field(default_factory=time.monotonic)
    status:            int   = 0    # trainer_pb2.WorkerStatus value
    shard_indices:     list  = field(default_factory=list)
    local_batch_size:  int   = 32
    assigned:          bool  = False
    last_loss:         float = 0.0
    steps:             int   = 0

    cmd_queue:        asyncio.Queue = field(default_factory=asyncio.Queue)
    assignment_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Set by _assign_late_worker before queuing RESHARD; cleared in Heartbeat handler.
    pending_reshard: Optional[trainer_pb2.ShardAssignment] = None

    @property
    def is_alive(self) -> bool:
        return (time.monotonic() - self.last_heartbeat) < self.heartbeat_timeout


def infer_device(w: WorkerState) -> str:
    """Return the best device string for a worker based on its hardware summary."""
    if "CUDA" in w.accel_summary:
        return "cuda:0"
    if "MPS" in w.accel_summary:
        return "mps"
    return "cpu"
