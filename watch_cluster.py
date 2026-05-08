"""
watch_cluster.py  —  Live cluster monitor + worker admission control

Usage:
    python3 watch_cluster.py [--leader leader-macbook-pro.taila5426e.ts.net] [--port 50051]

Displays live status and accepts commands:
    admit <id>   — admit a specific pending worker into the running training
    admit all    — admit every pending worker at once
    quit         — exit
"""
from __future__ import annotations

import argparse
import os
import select
import sys
import time

import grpc

from proto import trainer_pb2, trainer_pb2_grpc

_DIV = "─" * 88


def _clear() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


def _status_table(resp) -> str:
    workers = resp.workers
    if not workers:
        return "  (no active workers)\n"

    lines = [_DIV]
    lines.append(
        f"  {'ID':<10} {'HOSTNAME':<28} {'SCORE':>8} "
        f"{'SHARD':>7} {'STATUS':<12} ACCEL"
    )
    lines.append(_DIV)

    status_names = {0: "idle", 1: "training", 2: "downloading", 3: "error"}
    for w in workers:
        age    = int(time.time()) - w.last_seen_ts
        alive  = "alive" if age < 30 else f"silent {age}s"
        status = status_names.get(w.status, str(w.status))
        lines.append(
            f"  {w.worker_id:<10} {w.hostname:<28} {w.score:>8.1f} "
            f"{w.shard_pct:>6.1f}%  {alive:<12}"
        )
    lines.append(_DIV)
    lines.append(
        f"  Step={resp.global_step:>6,}  "
        f"Epoch={resp.current_epoch:>3}  "
        f"Loss={resp.global_loss:.4f}  "
        f"Active workers={len(workers)}"
    )
    return "\n".join(lines)


def _pending_table(pending) -> str:
    if not pending:
        return ""
    lines = ["", f"  {'PENDING WORKERS — awaiting admission':^86}", _DIV]
    lines.append(
        f"  {'ID':<10} {'HOSTNAME':<28} {'SCORE':>8} {'WAITING':>9}  ACCEL"
    )
    lines.append(_DIV)
    for w in pending:
        wait = f"{w.waiting_seconds}s"
        lines.append(
            f"  {w.worker_id:<10} {w.hostname:<28} {w.score:>8.1f} {wait:>9}  {w.accel_summary}"
        )
    lines.append(_DIV)
    ids = " ".join(w.worker_id for w in pending)
    lines.append(f"  To admit: admit all  |  admit {pending[0].worker_id}  ...")
    return "\n".join(lines)


def _process_command(line: str, stub: trainer_pb2_grpc.TrainerServiceStub) -> None:
    parts = line.strip().split()
    if not parts:
        return
    cmd = parts[0].lower()

    if cmd == "admit":
        # "admit all" or "admit <id> [id ...]"
        if not parts[1:] or parts[1].lower() == "all":
            ids = []   # empty = admit all
            label = "all pending"
        else:
            ids   = parts[1:]
            label = ", ".join(ids)

        try:
            resp = stub.AdmitWorkers(
                trainer_pb2.AdmitWorkersRequest(worker_ids=ids),
                timeout=5.0,
            )
            if resp.admitted_count:
                print(f"\n  Admitted {resp.admitted_count} worker(s): {list(resp.admitted_ids)}")
            else:
                print(f"\n  No workers admitted (requested: {label})")
            if resp.not_found_ids:
                print(f"  Not found / already active: {list(resp.not_found_ids)}")
        except grpc.RpcError as e:
            print(f"\n  AdmitWorkers failed: {e.code().name} — {e.details()}")

    elif cmd in ("quit", "q", "exit"):
        print("\nBye.")
        sys.exit(0)

    else:
        print(f"\n  Unknown command: {cmd!r}")
        print("  Commands: admit [id ...] | admit all | quit")


def main() -> None:
    p = argparse.ArgumentParser(description="Live cluster monitor with worker admission")
    p.add_argument("--leader",   default="leader-macbook-pro.taila5426e.ts.net")
    p.add_argument("--port",     type=int,   default=50051)
    p.add_argument("--interval", type=float, default=3.0,
                   help="Refresh interval in seconds (default 3)")
    cfg = p.parse_args()

    addr = f"{cfg.leader}:{cfg.port}"
    ch   = grpc.insecure_channel(addr, options=[
        ("grpc.max_receive_message_length", 4 * 1024 * 1024),
    ])
    stub = trainer_pb2_grpc.TrainerServiceStub(ch)

    print(f"Watching {addr}  (Ctrl-C or 'quit' to exit)\n")

    while True:
        try:
            resp = stub.GetClusterStatus(
                trainer_pb2.ClusterStatusRequest(), timeout=5.0
            )
            _clear()
            print(f"  Cluster  [{time.strftime('%H:%M:%S')}]  {addr}\n")
            print(_status_table(resp))
            if resp.pending_workers:
                print(_pending_table(resp.pending_workers))
        except grpc.RpcError as e:
            _clear()
            print(f"  [{time.strftime('%H:%M:%S')}] Leader unreachable: {e.code().name}")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        # Non-blocking input: wait up to --interval seconds for a command.
        # Falls back to dumb sleep on Windows (no select on stdin there).
        print(f"\n  cmd> ", end="", flush=True)
        try:
            if sys.platform == "win32":
                time.sleep(cfg.interval)
            else:
                ready, _, _ = select.select([sys.stdin], [], [], cfg.interval)
                if ready:
                    line = sys.stdin.readline()
                    _process_command(line, stub)
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
