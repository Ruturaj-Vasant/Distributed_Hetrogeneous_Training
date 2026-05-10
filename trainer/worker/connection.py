"""
trainer/worker/connection.py — gRPC channel factories and leader registration.
"""
from __future__ import annotations
import asyncio
import logging

import grpc
from grpc import aio

from proto import trainer_pb2, trainer_pb2_grpc
from trainer.core.errors import LeaderConnectionError, fmt_grpc_error
from trainer.core.logging import get as _get_log
from trainer.worker.proto_helpers import hw_to_proto

log = _get_log("worker")

# ResNet-101 state dict ≈ 170 MB; allow 256 MB messages on both sides.
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length",    256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]


def sync_channel(host: str, port: int) -> grpc.Channel:
    """Synchronous channel — used in the training thread."""
    return grpc.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)


def async_channel(host: str, port: int) -> aio.Channel:
    """Async channel — used for heartbeat and GetAssignment coroutines."""
    return aio.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)


async def connect_and_register(
    cfg,
    hw,
    bm,
) -> tuple[
    aio.Channel,
    trainer_pb2_grpc.TrainerServiceStub,
    grpc.Channel,
    trainer_pb2_grpc.TrainerServiceStub,
    trainer_pb2.RegisterResponse,
]:
    """
    Open fresh channels and register with the leader, retrying transient
    Tailscale/gRPC startup failures instead of crashing on the first error.

    Raises LeaderConnectionError after cfg.connect_retries failed attempts.
    """
    addr = f"{cfg.leader}:{cfg.port}"
    last_error = "unknown error"

    for attempt in range(1, cfg.connect_retries + 1):
        async_ch   = async_channel(cfg.leader, cfg.port)
        async_stub = trainer_pb2_grpc.TrainerServiceStub(async_ch)
        sync_ch    = sync_channel(cfg.leader, cfg.port)
        sync_stub  = trainer_pb2_grpc.TrainerServiceStub(sync_ch)

        try:
            log.info(
                f"Connecting to leader at {addr} "
                f"(attempt {attempt}/{cfg.connect_retries}) …"
            )
            await asyncio.wait_for(
                async_ch.channel_ready(),
                timeout=cfg.connect_timeout,
            )
            reg = await async_stub.Register(
                trainer_pb2.RegisterRequest(
                    hw_info=hw_to_proto(hw),
                    benchmark=trainer_pb2.BenchmarkResult(
                        score=bm.score,
                        forward_ms=bm.forward_ms,
                        memory_free_gb=bm.memory_free_gb,
                    ),
                ),
                timeout=cfg.rpc_timeout,
                wait_for_ready=True,
            )
            return async_ch, async_stub, sync_ch, sync_stub, reg

        except (asyncio.TimeoutError, grpc.aio.AioRpcError) as exc:
            last_error = fmt_grpc_error(exc)
            await async_ch.close()
            sync_ch.close()

            if attempt >= cfg.connect_retries:
                break

            delay = min(cfg.retry_backoff * attempt, cfg.max_retry_backoff)
            log.warning(
                f"Leader connection failed: {last_error}; "
                f"retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise LeaderConnectionError(
        f"Could not register with leader at {addr} after "
        f"{cfg.connect_retries} attempt(s): {last_error}"
    )
