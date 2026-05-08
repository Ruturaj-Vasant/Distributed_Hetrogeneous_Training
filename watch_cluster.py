"""
watch_cluster.py  —  Live cluster status monitor

Usage:
    python3 watch_cluster.py [--leader leader-macbook-pro.taila5426e.ts.net] [--port 50051] [--interval 3]

Polls GetClusterStatus every --interval seconds and prints a live table.
Run in a separate terminal while leader and workers are active.
"""

import argparse
import os
import sys
import time

import grpc

from proto import trainer_pb2, trainer_pb2_grpc


def _clear():
    os.system("cls" if sys.platform == "win32" else "clear")


def _status_table(resp) -> str:
    workers = resp.workers
    if not workers:
        return "  (no workers connected)\n"

    total_pct = sum(w.shard_pct for w in workers)
    div = "─" * 88
    lines = [div]
    lines.append(
        f"  {'ID':<10} {'HOSTNAME':<28} {'SCORE':>8} "
        f"{'SHARD':>7} {'STATUS':<10}  ACCEL"
    )
    lines.append(div)

    status_names = {0: "idle", 1: "training", 2: "idle"}

    for w in workers:
        age = int(time.time()) - w.last_seen_ts
        alive = "alive" if age < 30 else f"silent {age}s"
        status = status_names.get(w.status, str(w.status))
        lines.append(
            f"  {w.worker_id:<10} {w.hostname:<28} {w.score:>8.1f} "
            f"{w.shard_pct:>6.1f}%  {alive:<10}  "
        )
    lines.append(div)
    lines.append(
        f"  Step={resp.global_step:>6,}  "
        f"Epoch={resp.current_epoch:>3}  "
        f"Loss={resp.global_loss:.4f}  "
        f"Workers={len(workers)}"
    )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Live cluster status watcher")
    p.add_argument("--leader",   default="leader-macbook-pro.taila5426e.ts.net")
    p.add_argument("--port",     type=int, default=50051)
    p.add_argument("--interval", type=float, default=3.0,
                   help="Poll interval in seconds (default 3)")
    cfg = p.parse_args()

    addr = f"{cfg.leader}:{cfg.port}"
    ch   = grpc.insecure_channel(addr)
    stub = trainer_pb2_grpc.TrainerServiceStub(ch)

    print(f"Watching {addr}  (Ctrl-C to quit)\n")

    while True:
        try:
            resp = stub.GetClusterStatus(
                trainer_pb2.ClusterStatusRequest(),
                timeout=5.0,
            )
            _clear()
            print(f"  Cluster status  [{time.strftime('%H:%M:%S')}]  — {addr}\n")
            print(_status_table(resp))
        except grpc.RpcError as e:
            _clear()
            print(f"  [{time.strftime('%H:%M:%S')}] Leader not reachable: {e.code().name}")
        except KeyboardInterrupt:
            print("\nBye.")
            break
        time.sleep(cfg.interval)


if __name__ == "__main__":
    main()
