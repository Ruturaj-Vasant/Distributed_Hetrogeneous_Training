"""
watch_cluster.py — Live cluster monitor + worker admission control

Usage:
    python3 watch_cluster.py [--leader <host>] [--port 50051] [--interval 5]

Two-panel display:
    Top:    Admitted workers (active training participants)
    Bottom: Pending workers (waiting for admission)

Commands at the cmd> prompt:
    admit <id> [id ...]  — admit specific pending worker(s)
    admit all            — admit every pending worker
    r / refresh          — force a display refresh
    quit / q             — exit
"""
from __future__ import annotations

import argparse
import queue
import select
import sys
import threading
import time

import grpc

from proto import trainer_pb2, trainer_pb2_grpc

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

# ── Constants ─────────────────────────────────────────────────────────────────

_ALIVE_SECS  = 30     # seconds without heartbeat before worker is "disconnected"
_PAD_ADMITTED = 4     # minimum rows in the admitted table (keeps height stable)
_PAD_PENDING  = 3     # minimum rows in the pending table

# ── Rich helpers ──────────────────────────────────────────────────────────────

def _age_cell(last_seen_ts: int) -> Text:
    age = max(0, int(time.time()) - last_seen_ts)
    if age < _ALIVE_SECS:
        return Text(f"{age}s", style="green")
    if age < 120:
        return Text(f"silent {age}s", style="yellow")
    return Text(f"DEAD {age}s", style="red bold")


def _status_cell(status: int, last_seen_ts: int) -> Text:
    age = max(0, int(time.time()) - last_seen_ts)
    if age >= _ALIVE_SECS:
        return Text("disconnected", style="red bold")
    label, style = {
        0: ("idle",        "white"),
        1: ("training",    "green bold"),
        2: ("downloading", "yellow"),
        3: ("error",       "red"),
    }.get(status, (str(status), "white"))
    return Text(label, style=style)


# ── Table builders ────────────────────────────────────────────────────────────

def _admitted_table(workers) -> Table:
    t = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        expand=True,
        show_edge=False,
        padding=(0, 1),
    )
    t.add_column("Worker ID",  style="cyan",  no_wrap=True, min_width=10)
    t.add_column("Hostname",   no_wrap=True,  min_width=22)
    t.add_column("Score",      justify="right", min_width=6)
    t.add_column("Shard %",    justify="right", min_width=7)
    t.add_column("Status",     min_width=14)
    t.add_column("Last Seen",  justify="right", min_width=11)

    if not workers:
        t.add_row("[dim]—[/dim]", "[dim]no admitted workers yet[/dim]",
                  "", "", "", "")
        for _ in range(_PAD_ADMITTED - 1):
            t.add_row("", "", "", "", "", "")
    else:
        for w in workers:
            t.add_row(
                w.worker_id,
                w.hostname,
                f"{w.score:.0f}",
                f"{w.shard_pct:.1f}%",
                _status_cell(w.status, w.last_seen_ts),
                _age_cell(w.last_seen_ts),
            )
        for _ in range(max(0, _PAD_ADMITTED - len(workers))):
            t.add_row("", "", "", "", "", "")
    return t


def _pending_table(pending) -> Table:
    t = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold yellow",
        expand=True,
        show_edge=False,
        padding=(0, 1),
    )
    t.add_column("Worker ID", style="cyan", no_wrap=True, min_width=10)
    t.add_column("Hostname",  no_wrap=True, min_width=22)
    t.add_column("Score",     justify="right", min_width=6)
    t.add_column("Waiting",   justify="right", min_width=8)
    t.add_column("Accel",     min_width=20)

    if not pending:
        t.add_row("[dim]—[/dim]", "[dim]no pending workers[/dim]",
                  "", "", "")
        for _ in range(_PAD_PENDING - 1):
            t.add_row("", "", "", "", "")
    else:
        for w in pending:
            t.add_row(
                w.worker_id,
                w.hostname,
                f"{w.score:.0f}",
                f"{w.waiting_seconds}s",
                w.accel_summary,
            )
        for _ in range(max(0, _PAD_PENDING - len(pending))):
            t.add_row("", "", "", "", "")
    return t


# ── Full renderable ───────────────────────────────────────────────────────────

def _build_display(resp, addr: str) -> Group:
    n_admitted = len(resp.workers)
    n_pending  = len(resp.pending_workers)

    admitted_panel = Panel(
        _admitted_table(resp.workers),
        title=f"[cyan bold] Admitted Workers ({n_admitted}) [/cyan bold]",
        border_style="cyan",
    )

    pending_panel = Panel(
        _pending_table(resp.pending_workers),
        title=f"[yellow bold] Pending Workers ({n_pending}) [/yellow bold]",
        border_style="yellow",
    )

    # Status bar
    if resp.training_started:
        status = (
            f"[green]● TRAINING[/green]  "
            f"Step [bold]{resp.global_step:,}[/bold]  "
            f"Epoch [bold]{resp.current_epoch}[/bold]  "
            f"Loss [bold]{resp.global_loss:.4f}[/bold]"
        )
    else:
        status = "[yellow]● WAITING — type [bold]start[/bold] on the leader terminal[/yellow]"

    status_line = Text.from_markup(
        f"  {status}   [dim]{addr}   {time.strftime('%H:%M:%S')}[/dim]"
    )

    # Quick-admit commands so user never has to copy-paste IDs manually
    if resp.pending_workers:
        parts = ["[bold yellow]admit all[/bold yellow]"] + [
            f"[yellow]admit {w.worker_id}[/yellow]"
            for w in resp.pending_workers
        ]
        hint = "  Quick:  " + "   |   ".join(parts)
    elif resp.training_started and n_admitted == 0:
        hint = "  [red]No active workers — training may be stalled.[/red]"
    else:
        hint = ""

    hint_line = Text.from_markup(hint) if hint else Text("")

    cmd_hint = Text.from_markup(
        "  [dim]admit <id> | admit all | r = refresh | q = quit[/dim]"
    )

    return Group(
        admitted_panel,
        pending_panel,
        status_line,
        hint_line,
        cmd_hint,
        Text(""),   # spacer so cmd> prompt below has breathing room
    )


# ── Command processing ────────────────────────────────────────────────────────

def _process_command(
    line: str,
    stub: trainer_pb2_grpc.TrainerServiceStub,
    console: Console,
) -> None:
    parts = line.strip().split()
    if not parts:
        return
    cmd = parts[0].lower()

    if cmd == "admit":
        if not parts[1:] or parts[1].lower() == "all":
            ids   = []
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
                console.print(
                    f"  [green]✓ Admitted {resp.admitted_count} worker(s):"
                    f" {list(resp.admitted_ids)}[/green]"
                )
            else:
                console.print(f"  [yellow]No workers admitted (requested: {label})[/yellow]")
            if resp.not_found_ids:
                console.print(
                    f"  [yellow]Not found / already active:"
                    f" {list(resp.not_found_ids)}[/yellow]"
                )
        except grpc.RpcError as e:
            console.print(f"  [red]AdmitWorkers failed: {e.code().name} — {e.details()}[/red]")

    elif cmd in ("quit", "q", "exit"):
        console.print("\n  Bye.")
        sys.exit(0)

    elif cmd in ("r", "refresh"):
        pass  # caller will refresh on next iteration

    else:
        console.print(f"  [red]Unknown command:[/red] {cmd!r}")
        console.print("  Commands: admit [id …] | admit all | r | quit")


# ── Input thread ──────────────────────────────────────────────────────────────

def _start_input_thread(cmd_q: queue.Queue) -> None:
    """Daemon thread: reads lines from stdin and puts them into cmd_q."""
    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd_q.put(line.strip())
            except Exception:
                break
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


# ── Main ──────────────────────────────────────────────────────────────────────

def _unreachable(addr: str) -> Text:
    return Text.from_markup(
        f"  [red]Leader unreachable[/red]  [dim]{addr}  {time.strftime('%H:%M:%S')}[/dim]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Live cluster monitor with worker admission")
    p.add_argument("--leader",   default="leader-macbook-pro.taila5426e.ts.net")
    p.add_argument("--port",     type=int,   default=50051)
    p.add_argument("--interval", type=float, default=5.0,
                   help="Auto-refresh interval in seconds (default 5)")
    cfg = p.parse_args()

    addr    = f"{cfg.leader}:{cfg.port}"
    ch      = grpc.insecure_channel(addr, options=[
        ("grpc.max_receive_message_length", 4 * 1024 * 1024),
    ])
    stub    = trainer_pb2_grpc.TrainerServiceStub(ch)
    console = Console()
    cmd_q: queue.Queue[str] = queue.Queue()

    _start_input_thread(cmd_q)

    console.print(
        f"\n  Watching [bold]{addr}[/bold]\n"
        f"  [dim]Type commands below the display  "
        f"(admit all | admit <id> | r = refresh | q = quit)[/dim]\n"
    )

    # Fetch initial status
    try:
        resp = stub.GetClusterStatus(trainer_pb2.ClusterStatusRequest(), timeout=5.0)
        initial = _build_display(resp, addr)
    except grpc.RpcError:
        initial = _unreachable(addr)

    with Live(
        initial,
        console=console,
        auto_refresh=False,      # we call live.refresh() manually — no cursor flicker while typing
        refresh_per_second=4,
        transient=False,
        vertical_overflow="visible",
    ) as live:

        last_refresh = time.monotonic()

        while True:
            # ── Drain command queue (non-blocking) ────────────────────────
            processed_cmd = False
            while True:
                try:
                    line = cmd_q.get_nowait()
                except queue.Empty:
                    break

                if not line:
                    continue
                if line.lower() in ("q", "quit", "exit"):
                    live.stop()
                    console.print("\n  Bye.")
                    sys.exit(0)

                _process_command(line, stub, console)
                processed_cmd = True

            # ── Refresh display (after command or on interval) ────────────
            now = time.monotonic()
            if processed_cmd or (now - last_refresh >= cfg.interval):
                try:
                    resp = stub.GetClusterStatus(
                        trainer_pb2.ClusterStatusRequest(), timeout=5.0
                    )
                    live.update(_build_display(resp, addr))
                except grpc.RpcError as e:
                    live.update(_unreachable(addr))
                live.refresh()
                last_refresh = time.monotonic()

            # Sleep until next check — wake early if a command arrives
            try:
                line = cmd_q.get(timeout=min(0.25, cfg.interval))
                cmd_q.put(line)   # put back so it's processed on next iteration
            except queue.Empty:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Bye.")
