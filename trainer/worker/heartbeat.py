"""
trainer/worker/heartbeat.py — Shared state bulletin board + async heartbeat stream.
"""
from __future__ import annotations
import asyncio
import time
from typing import Optional

from proto import trainer_pb2, trainer_pb2_grpc
from trainer.core.logging import get as _get_log

log = _get_log("worker")


class WorkerSharedState:
    """
    Thread-safe bulletin board between the heartbeat coroutine and the
    training thread. Python's GIL makes simple attribute reads/writes safe
    for the primitive types used here (int, float, bool).
    """
    def __init__(self) -> None:
        self.status:              int   = trainer_pb2.IDLE
        self.loss:                float = -1.0
        self.steps:               int   = 0
        self.paused:              bool  = False
        self.stop:                bool  = False
        self.leader_disconnected: bool  = False
        # Populated by heartbeat_loop on RESHARD; cleared by runner after reload.
        self.reshard_indices:    Optional[list] = None
        self.reshard_batch_size: int            = 0
        # Tracks the epoch in progress — lets the worker continue from the right
        # epoch after a reshard rather than restarting from epoch 0.
        self.epoch_completed:    int            = 0


async def heartbeat_loop(
    stub:      trainer_pb2_grpc.TrainerServiceStub,
    worker_id: str,
    shared:    WorkerSharedState,
) -> None:
    """
    Bidirectional heartbeat stream.
    Worker sends status every 2 s; leader replies with a command.
    Runs as an asyncio background task for the worker's entire lifetime.
    Stopped only by task cancellation (runner.py finally block) or a STOP
    command — NOT by shared.stop, so reshard does not kill the stream.
    """
    async def _requests():
        # Runs until this coroutine is cancelled — never exits on shared.stop
        # so that a RESHARD (which sets shared.stop to exit the training thread)
        # does not also kill the heartbeat stream.
        while True:
            yield trainer_pb2.HeartbeatRequest(
                worker_id       = worker_id,
                status          = shared.status,
                current_loss    = shared.loss,
                steps_completed = shared.steps,
                timestamp_utc   = int(time.time()),
            )
            await asyncio.sleep(2)

    try:
        async for resp in stub.Heartbeat(_requests()):
            cmd = resp.command
            if cmd == trainer_pb2.HeartbeatResponse.STOP:
                log.warning("Leader sent STOP — halting training.")
                shared.stop = True
                return
            elif cmd == trainer_pb2.HeartbeatResponse.PAUSE:
                log.info("Leader sent PAUSE.")
                shared.paused = True
            elif cmd == trainer_pb2.HeartbeatResponse.CONTINUE:
                if shared.paused:
                    log.info("Leader sent CONTINUE — resuming.")
                shared.paused = False
            elif cmd == trainer_pb2.HeartbeatResponse.RESHARD:
                ns = resp.new_shard
                log.info(
                    f"Leader sent RESHARD — "
                    f"{len(ns.indices):,} new samples  batch={ns.local_batch_size}"
                )
                shared.reshard_indices    = list(ns.indices)
                shared.reshard_batch_size = ns.local_batch_size
                # Signal the training thread to exit and rebuild the loader.
                # Do NOT return here — the heartbeat stream must stay alive
                # so the leader keeps seeing this worker during reshard.
                shared.stop = True
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error(f"Heartbeat stream lost: {type(e).__name__}: {e}")
        shared.leader_disconnected = True
        shared.stop = True
