"""
trainer/leader/server.py — gRPC server startup, interactive CLI, port guard.
"""
from __future__ import annotations
import asyncio
import os
import signal
import time

from grpc import aio

from proto import trainer_pb2, trainer_pb2_grpc
from trainer.leader.service import LeaderService
from trainer.core.logging import setup as _setup_log
from trainer.dashboard import serve as _dashboard_serve

log = _setup_log("leader")

_DEFAULT_HEARTBEAT_TIMEOUT        = 60.0
_DEFAULT_HEARTBEAT_CHECK_INTERVAL =  5.0
_DEFAULT_GRAD_SYNC_TIMEOUT        = 120.0


# ── Port conflict guard ───────────────────────────────────────────────────────

def kill_existing_leader(port: int) -> None:
    """
    Kill any process already listening on *port* so only one leader owns it.
    Uses lsof (macOS/Linux). Silently skips on Windows or if lsof is absent.
    """
    import subprocess, shutil
    if not shutil.which("lsof"):
        return
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return

    killed = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass

    if killed:
        log.info(f"Stopped existing leader process(es) on port {port}: {killed}")
        time.sleep(0.5)


# ── Interactive CLI ───────────────────────────────────────────────────────────

async def cli_loop(service: LeaderService) -> None:
    loop = asyncio.get_event_loop()
    print()
    print("  Commands:  admit [id ...] | admit all | start | status | reset | quit")
    print()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("leader> "))
        except EOFError:
            break
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()

        if cmd == "start":
            await service.start_training()
        elif cmd == "status":
            await service._print_status()
        elif cmd == "reset":
            await service.reset_training()
        elif cmd == "admit":
            ids = parts[1:]
            if ids == ["all"]:
                ids = []
            resp = await service.AdmitWorkers(
                trainer_pb2.AdmitWorkersRequest(worker_ids=ids), context=None
            )
            if resp.admitted_count:
                print(f"  Admitted {resp.admitted_count}: {list(resp.admitted_ids)}")
            else:
                print("  No workers admitted.")
            if resp.not_found_ids:
                print(f"  Not found / already active: {list(resp.not_found_ids)}")
        elif cmd in ("quit", "exit", "q"):
            log.info("Shutting down.")
            return
        else:
            print(f"  Unknown command: {cmd!r}")
            print("  Try: admit [id ...] | admit all | start | status | reset | quit")


# ── Server startup ────────────────────────────────────────────────────────────

async def main(cfg) -> None:
    kill_existing_leader(cfg.port)

    service = LeaderService(cfg)
    _MB = 1024 * 1024
    server = aio.server(options=[
        ("grpc.max_receive_message_length", 256 * _MB),
        ("grpc.max_send_message_length",    256 * _MB),
        ("grpc.so_reuseport", 0),
    ])
    trainer_pb2_grpc.add_TrainerServiceServicer_to_server(service, server)

    listen_addr = f"0.0.0.0:{cfg.port}"
    server.add_insecure_port(listen_addr)
    await server.start()

    log.info(f"gRPC server listening on {listen_addr}")
    log.info(f"Tailscale DNS  : leader-macbook-pro.taila5426e.ts.net:{cfg.port}")
    log.info(f"Min workers    : {cfg.min_workers}")
    log.info(f"Auto-start     : {cfg.auto_start}")
    log.info(f"Device (leader): {service._device}")
    log.info(
        f"Timeouts       : heartbeat={cfg.heartbeat_timeout}s  "
        f"check={cfg.heartbeat_check_interval}s  "
        f"grad_sync={cfg.grad_sync_timeout}s"
    )

    asyncio.create_task(service.heartbeat_monitor())
    asyncio.create_task(service.status_printer())

    if not getattr(cfg, "no_dashboard", False):
        dashboard_port = getattr(cfg, "dashboard_port", 8080)
        await _dashboard_serve(service, port=dashboard_port)
        log.info(f"Dashboard        : http://localhost:{dashboard_port}")

    if os.isatty(0):
        try:
            await cli_loop(service)
        finally:
            if service._recorder is not None:
                run_dir = service._recorder.close()
                log.info(f"Run saved to {run_dir}")
            await server.stop(grace=5)
    else:
        log.info("Non-interactive mode — Ctrl-C or SIGTERM to stop.")
        try:
            await server.wait_for_termination()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            if service._recorder is not None:
                run_dir = service._recorder.close()
                log.info(f"Run saved to {run_dir}")
            await server.stop(grace=5)
