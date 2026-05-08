"""
leader.py  —  gRPC parameter server + cluster orchestrator

Usage:
    python3 leader.py [--port 50051] [--min-workers 1] [--epochs 90]

Interactive commands (after startup):
    start   — begin training with all currently alive workers
    status  — print live cluster table
    quit    — graceful shutdown

Worker flow (handled here):
    1. Worker calls Register()       → assigned a worker_id
    2. Worker calls GetAssignment()  → BLOCKS until "start" is typed
    3. Leader sends shard + config + initial model weights
    4. Worker trains; calls ExchangeGradients() each step
    5. Leader aggregates (weighted by batch count), returns weight delta
"""

import argparse
import asyncio
import io
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import grpc
from grpc import aio

from proto import trainer_pb2, trainer_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [leader] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("leader")

# ── Tunables ──────────────────────────────────────────────────────────────────

STATUS_INTERVAL     = 10.0      # seconds between auto-printed cluster tables
TINY_IMAGENET_TRAIN = 100_000   # total training samples in Tiny ImageNet-200
# Heartbeat / grad-sync defaults — overridden by CLI flags at runtime
_DEFAULT_HEARTBEAT_TIMEOUT       = 60.0
_DEFAULT_HEARTBEAT_CHECK_INTERVAL =  5.0
_DEFAULT_GRAD_SYNC_TIMEOUT       = 120.0


# ── Per-worker state ──────────────────────────────────────────────────────────

@dataclass
class WorkerState:
    worker_id:         str
    hostname:          str
    os_name:           str
    score:             float
    cpu_cores:         int
    ram_gb:            float
    accel_summary:     str          # human-readable: "CUDA:RTX3060@6GB | ..."
    heartbeat_timeout: float = _DEFAULT_HEARTBEAT_TIMEOUT
    registered_at:     float = field(default_factory=time.monotonic)
    last_heartbeat:    float = field(default_factory=time.monotonic)
    status:            int   = 0    # trainer_pb2.WorkerStatus value
    shard_indices:     list  = field(default_factory=list)
    local_batch_size:  int   = 32
    assigned:          bool  = False

    # asyncio primitives — created fresh per worker
    cmd_queue:        asyncio.Queue = field(default_factory=asyncio.Queue)
    assignment_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_alive(self) -> bool:
        return (time.monotonic() - self.last_heartbeat) < self.heartbeat_timeout


# ── gRPC service ──────────────────────────────────────────────────────────────

class LeaderService(trainer_pb2_grpc.TrainerServiceServicer):

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()

        # Training flags
        self._training_started = asyncio.Event()
        self._training_cfg: Optional[trainer_pb2.TrainingConfig] = None

        # Model lives on the leader (parameter server role)
        self._device     = self._pick_device()
        self._model:     Optional[nn.Module]             = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._prev_state: dict[str, torch.Tensor]        = {}

        # Metrics
        self._global_step   = 0
        self._current_epoch = 0
        self._global_loss   = 0.0

        # Gradient synchronisation across workers
        # asyncio.Condition protects _pending_grads and _grad_round.
        # Workers block in wait_for() until the round increments.
        self._grad_cond       = asyncio.Condition()
        self._pending_grads: dict[str, trainer_pb2.GradientPush] = {}
        self._grad_round      = 0
        self._grad_round_start: float = 0.0   # monotonic time when first grad arrived this round
        self._last_payload: bytes = b""

    # ── RPC: Register ─────────────────────────────────────────────────────────

    async def Register(
        self,
        request: trainer_pb2.RegisterRequest,
        context,
    ) -> trainer_pb2.RegisterResponse:

        if self._training_started.is_set():
            # Late joiners not yet supported — could be added as a reconfigure
            return trainer_pb2.RegisterResponse(
                accepted=False,
                reject_reason="Training already in progress.",
            )

        hw  = request.hw_info
        bm  = request.benchmark
        wid = str(uuid.uuid4())[:8]

        accel_parts = []
        for a in hw.accelerators:
            if a.type == trainer_pb2.AcceleratorInfo.CUDA:
                accel_parts.append(f"CUDA:{a.name}@{a.vram_gb:.0f}GB")
            elif a.type == trainer_pb2.AcceleratorInfo.MPS:
                accel_parts.append(f"MPS:{a.name}@{a.gpu_cores}cores")
        accel_summary = " | ".join(accel_parts) or "CPU-only"

        state = WorkerState(
            worker_id         = wid,
            hostname          = hw.hostname,
            os_name           = hw.os,
            score             = bm.score,
            cpu_cores         = hw.cpu_cores,
            ram_gb            = hw.ram_gb,
            accel_summary     = accel_summary,
            heartbeat_timeout = self.cfg.heartbeat_timeout,
        )

        async with self._lock:
            self._workers[wid] = state

        log.info(
            f"[+] {wid}  host={hw.hostname:<22}  "
            f"score={bm.score:>8.1f}  bench={bm.forward_ms:.1f}ms  {accel_summary}"
        )

        # Auto-start once threshold is met (if flag is set)
        if self.cfg.auto_start and not self._training_started.is_set():
            alive = await self._count_alive()
            if alive >= self.cfg.min_workers:
                asyncio.create_task(self.start_training())

        return trainer_pb2.RegisterResponse(worker_id=wid, accepted=True)

    # ── RPC: Heartbeat (bidirectional stream) ─────────────────────────────────

    async def Heartbeat(self, request_iterator, context):
        wid = None
        try:
            async for req in request_iterator:
                wid = req.worker_id

                async with self._lock:
                    w = self._workers.get(wid)
                if w is None:
                    await context.abort(
                        grpc.StatusCode.NOT_FOUND, f"Unknown worker {wid}"
                    )
                    return

                w.last_heartbeat = time.monotonic()
                w.status         = req.status

                # Drain one queued command (CONTINUE if nothing pending)
                try:
                    cmd = w.cmd_queue.get_nowait()
                except asyncio.QueueEmpty:
                    cmd = trainer_pb2.HeartbeatResponse.CONTINUE

                yield trainer_pb2.HeartbeatResponse(command=cmd)

        except grpc.RpcError:
            pass
        finally:
            if wid:
                log.warning(f"Heartbeat stream closed for {wid}")

    # ── RPC: GetAssignment (blocks until training starts) ─────────────────────

    async def GetAssignment(
        self,
        request: trainer_pb2.GetAssignmentRequest,
        context,
    ) -> trainer_pb2.GetAssignmentResponse:

        wid = request.worker_id
        async with self._lock:
            w = self._workers.get(wid)
        if w is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown {wid}")
            return

        log.info(f"Worker {wid} ({w.hostname}) waiting for assignment …")

        # Block here until start_training() sets the per-worker event
        await w.assignment_event.wait()

        # Re-read in case state changed during the wait
        async with self._lock:
            w = self._workers.get(wid)
        if w is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Worker {wid} was removed")
            return

        # Serialise full model weights in a thread (CPU-bound)
        model_bytes = await asyncio.get_event_loop().run_in_executor(
            None, self._serialize_full_model
        )
        w.assigned = True

        log.info(
            f"Assignment sent → {wid}  "
            f"samples={len(w.shard_indices):,}  "
            f"batch={w.local_batch_size}  "
            f"weights={len(model_bytes) // 1024:,}KB"
        )

        return trainer_pb2.GetAssignmentResponse(
            assignment=trainer_pb2.ShardAssignment(
                worker_id        = wid,
                indices          = w.shard_indices,
                local_batch_size = w.local_batch_size,
                primary_device   = _infer_device(w),
            ),
            config        = self._training_cfg,
            model_weights = model_bytes,
        )

    # ── RPC: ExchangeGradients ────────────────────────────────────────────────

    async def ExchangeGradients(
        self,
        request: trainer_pb2.GradientPush,
        context,
    ) -> trainer_pb2.WeightUpdate:
        """
        Synchronous gradient aggregation:
          - Workers push gradients for step N.
          - The last worker to arrive aggregates, updates the model,
            increments grad_round, and notifies all waiting coroutines.
          - All workers get the same weight delta back.

        Dead-worker recovery: if a worker is removed by heartbeat_monitor
        while others are waiting, monitor also acquires _grad_cond and
        triggers aggregation with fewer workers so no one blocks forever.
        """
        wid = request.worker_id

        # Gradient push proves the worker is alive — refresh liveness timestamp
        # so a slow training step doesn't trigger a false heartbeat timeout.
        async with self._lock:
            w = self._workers.get(wid)
            if w is not None:
                w.last_heartbeat = time.monotonic()

        round_before = self._grad_round

        async with self._grad_cond:
            if not self._pending_grads:
                self._grad_round_start = time.monotonic()   # first grad this round
            self._pending_grads[wid] = request
            n_alive    = await self._count_alive()
            n_received = len(self._pending_grads)

            log.debug(
                f"Grad push from {wid}  step={request.global_step}  "
                f"{n_received}/{n_alive} received"
            )

            if n_received >= n_alive:
                # Last (or only) worker — do the aggregation in a thread
                await asyncio.get_event_loop().run_in_executor(
                    None, self._aggregate_and_step
                )
                self._grad_round += 1
                self._grad_cond.notify_all()
            else:
                # Wait for remaining workers; Condition releases the lock
                await self._grad_cond.wait_for(
                    lambda: self._grad_round > round_before
                )

        return trainer_pb2.WeightUpdate(
            global_step        = self._global_step,
            is_full_state_dict = False,
            payload            = self._last_payload,
        )

    # ── RPC: GetClusterStatus ─────────────────────────────────────────────────

    async def GetClusterStatus(
        self,
        request: trainer_pb2.ClusterStatusRequest,
        context,
    ) -> trainer_pb2.ClusterStatusResponse:
        async with self._lock:
            workers = list(self._workers.values())

        total_samples = sum(len(w.shard_indices) for w in workers) or 1
        summaries = [
            trainer_pb2.WorkerSummary(
                worker_id    = w.worker_id,
                hostname     = w.hostname,
                score        = w.score,
                status       = w.status,
                shard_pct    = len(w.shard_indices) / total_samples * 100,
                last_seen_ts = int(w.last_heartbeat),
            )
            for w in workers
        ]
        return trainer_pb2.ClusterStatusResponse(
            workers       = summaries,
            global_step   = self._global_step,
            global_loss   = self._global_loss,
            current_epoch = self._current_epoch,
        )

    # ── Public: kick off training ─────────────────────────────────────────────

    async def start_training(self) -> None:
        if self._training_started.is_set():
            log.warning("Training is already running.")
            return

        async with self._lock:
            alive = [w for w in self._workers.values() if w.is_alive]

        if not alive:
            log.error("No alive workers — cannot start.")
            return

        # Score-proportional shard assignment
        indices = list(range(TINY_IMAGENET_TRAIN))
        log.info(f"Computing shards for {len(alive)} worker(s) …")
        self._compute_shards(alive, indices)

        # Initialise model and optimizer on leader (in thread — CPU-bound)
        log.info("Initialising ResNet-101 on leader …")
        await asyncio.get_event_loop().run_in_executor(None, self._init_model)

        self._training_cfg = trainer_pb2.TrainingConfig(
            model_name      = getattr(self.cfg, "model_name", "resnet101"),
            dataset         = "tiny_imagenet_200",
            num_classes     = self.cfg.num_classes,
            total_epochs    = self.cfg.epochs,
            base_lr         = self.cfg.lr,
            weight_decay    = self.cfg.weight_decay,
            gradient_topk_k = self.cfg.topk,
        )

        self._training_started.set()

        # Unblock every worker's GetAssignment call simultaneously
        for w in alive:
            w.assignment_event.set()

        log.info(
            f"Training started — assignments dispatched to {len(alive)} worker(s)."
        )

    # ── Background task: heartbeat monitor ───────────────────────────────────

    async def heartbeat_monitor(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.heartbeat_check_interval)

            async with self._lock:
                dead = [w for w in self._workers.values() if not w.is_alive]
                for w in dead:
                    log.warning(f"[-] Worker {w.worker_id} ({w.hostname}) timed out.")
                    del self._workers[w.worker_id]

            if self._training_started.is_set():
                async with self._grad_cond:
                    n_alive    = await self._count_alive()
                    n_received = len(self._pending_grads)

                    should_force = False
                    reason = ""

                    # Case 1: workers died and remaining grads are enough to proceed
                    if dead and 0 < n_received >= n_alive:
                        should_force = True
                        reason = f"{len(dead)} worker(s) removed ({n_received}/{n_alive} grads)"

                    # Case 2: grad round has been stuck longer than grad_sync_timeout
                    elif (n_received > 0
                          and self._grad_round_start > 0
                          and (time.monotonic() - self._grad_round_start)
                              > self.cfg.grad_sync_timeout):
                        should_force = True
                        age = time.monotonic() - self._grad_round_start
                        reason = (
                            f"sync timeout {age:.0f}s > {self.cfg.grad_sync_timeout}s "
                            f"({n_received}/{n_alive} grads)"
                        )

                    if should_force:
                        log.warning(f"Forcing gradient aggregation: {reason}")
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._aggregate_and_step
                        )
                        self._grad_round += 1
                        self._grad_cond.notify_all()

    # ── Background task: periodic status table ────────────────────────────────

    async def status_printer(self) -> None:
        while True:
            await asyncio.sleep(STATUS_INTERVAL)
            await self._print_status()

    async def _print_status(self) -> None:
        async with self._lock:
            workers = list(self._workers.values())

        if not workers:
            log.info("No workers connected.")
            return

        total = sum(len(w.shard_indices) for w in workers) or 1
        div   = "─" * 82
        print(div)
        print(
            f"  {'ID':<10} {'HOSTNAME':<22} {'SCORE':>8} "
            f"{'SHARD':>7} {'BATCH':>6}  {'STATUS':<8}  ACCEL"
        )
        print(div)
        for w in workers:
            pct   = len(w.shard_indices) / total * 100 if w.shard_indices else 0.0
            alive = "alive" if w.is_alive else "DEAD"
            print(
                f"  {w.worker_id:<10} {w.hostname:<22} {w.score:>8.1f} "
                f"{pct:>6.1f}% {w.local_batch_size:>6}  {alive:<8}  {w.accel_summary}"
            )
        print(div)
        print(
            f"  Step={self._global_step:>6}  "
            f"Epoch={self._current_epoch:>3}  "
            f"Loss={self._global_loss:.4f}  "
            f"Workers={len(workers)}"
        )

    # ── Private: device selection ─────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")

    async def _count_alive(self) -> int:
        async with self._lock:
            return sum(1 for w in self._workers.values() if w.is_alive)

    # ── Private: shard distribution ───────────────────────────────────────────

    def _compute_shards(
        self, workers: list[WorkerState], indices: list[int]
    ) -> None:
        total_score = sum(w.score for w in workers) or 1.0
        offset = 0
        for i, w in enumerate(workers):
            frac = w.score / total_score
            # Last worker gets remainder to avoid rounding gaps
            n = (
                len(indices) - offset
                if i == len(workers) - 1
                else int(frac * len(indices))
            )
            w.shard_indices    = indices[offset: offset + n]
            # Batch size scales with fraction but stays in [8, 64]
            w.local_batch_size = max(8, min(64, int(32 * frac * len(workers))))
            offset += n
            log.info(
                f"  {w.hostname:<22}  score={w.score:>8.1f}  "
                f"frac={frac * 100:>5.1f}%  samples={n:>6,}  "
                f"batch={w.local_batch_size}"
            )

    # ── Private: model initialisation ────────────────────────────────────────

    def _init_model(self) -> None:
        _MODELS = {
            "resnet18":  models.resnet18,
            "resnet50":  models.resnet50,
            "resnet101": models.resnet101,
        }
        model_fn = _MODELS.get(getattr(self.cfg, "model_name", "resnet101"), models.resnet101)
        self._model = model_fn(
            weights=None, num_classes=self.cfg.num_classes
        ).to(self._device)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=self.cfg.weight_decay,
        )
        self._prev_state = {
            k: v.cpu().clone() for k, v in self._model.state_dict().items()
        }
        mb   = sum(p.numel() * 4 for p in self._model.parameters()) / 1e6
        name = getattr(self.cfg, "model_name", "resnet101")
        log.info(f"{name} on {self._device}  ({mb:.0f} MB params)")

    # ── Private: gradient aggregation + optimizer step ────────────────────────

    def _aggregate_and_step(self) -> None:
        """
        Weighted gradient aggregation (runs in thread pool, safe because
        _grad_cond is held by the caller's asyncio coroutine — no other
        ExchangeGradients call can enter while we run).

        Weight of worker i = local_batch_count_i / total_samples
        This is identical to computing the gradient over the full
        union of batches as if they were processed by one model.
        """
        grads         = dict(self._pending_grads)
        total_samples = sum(g.local_batch_count for g in grads.values()) or 1

        self._optimizer.zero_grad()
        named_params = dict(self._model.named_parameters())

        for push in grads.values():
            weight = push.local_batch_count / total_samples
            for sg in push.gradients:
                param = named_params.get(sg.layer_name)
                if param is None:
                    continue
                shape     = list(sg.shape)
                n_elems   = 1
                for d in shape:
                    n_elems *= d
                flat = torch.zeros(n_elems, device=self._device)
                flat[list(sg.indices)] = torch.tensor(
                    list(sg.values), device=self._device
                )
                grad = flat.reshape(shape)
                if param.grad is None:
                    param.grad = grad * weight
                else:
                    param.grad.add_(grad, alpha=weight)

        self._optimizer.step()
        self._global_step += 1

        losses = [g.loss for g in grads.values() if g.loss > 0]
        if losses:
            self._global_loss = sum(losses) / len(losses)

        # Compute parameter delta (new - old) and serialise
        new_state  = self._model.state_dict()
        delta = {
            k: (v.cpu() - self._prev_state[k])
            for k, v in new_state.items()
            if k in self._prev_state
        }
        self._prev_state = {k: v.cpu().clone() for k, v in new_state.items()}

        buf = io.BytesIO()
        torch.save(delta, buf)
        self._last_payload = buf.getvalue()
        self._pending_grads.clear()

        if self._global_step % 10 == 0:
            log.info(
                f"Step {self._global_step:>6,}  "
                f"loss={self._global_loss:.4f}  "
                f"delta={len(self._last_payload) // 1024:,}KB  "
                f"workers={len(grads)}"
            )

        # Epoch tracking (approximate — assumes all workers finish same #steps)
        steps_per_epoch = max(1, TINY_IMAGENET_TRAIN // (32 * max(1, len(grads))))
        self._current_epoch = self._global_step // steps_per_epoch

    # ── Private: weight serialisation ─────────────────────────────────────────

    def _serialize_full_model(self) -> bytes:
        buf = io.BytesIO()
        torch.save(
            {k: v.cpu() for k, v in self._model.state_dict().items()},
            buf,
        )
        return buf.getvalue()


# ── Utility ───────────────────────────────────────────────────────────────────

def _infer_device(w: WorkerState) -> str:
    if "CUDA" in w.accel_summary:
        return "cuda:0"
    if "MPS" in w.accel_summary:
        return "mps"
    return "cpu"


# ── Interactive CLI ───────────────────────────────────────────────────────────

async def cli_loop(service: LeaderService) -> None:
    loop = asyncio.get_event_loop()
    print()
    print("  Commands:  start | status | quit")
    print()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("leader> "))
        except EOFError:
            break
        cmd = line.strip().lower()
        if cmd == "start":
            await service.start_training()
        elif cmd == "status":
            await service._print_status()
        elif cmd in ("quit", "exit", "q"):
            log.info("Shutting down.")
            sys.exit(0)
        elif cmd == "":
            pass
        else:
            print(f"  Unknown command: {cmd!r}")
            print("  Try: start | status | quit")


# ── Port conflict guard ───────────────────────────────────────────────────────

def _kill_existing_leader(port: int) -> None:
    """
    Find and kill any process already listening on *port* so that only one
    leader can own the port at a time.  Uses lsof (available on macOS/Linux).
    Silently does nothing on Windows or if lsof is not found.
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
        return  # no process found on that port

    killed = []
    for line in out.splitlines()[1:]:   # skip header
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue                    # don't kill ourselves
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass

    if killed:
        log.info(f"Stopped existing leader process(es) on port {port}: {killed}")
        time.sleep(0.5)                 # brief pause so the port is released


# ── Server startup ────────────────────────────────────────────────────────────

async def main(cfg: argparse.Namespace) -> None:
    _kill_existing_leader(cfg.port)

    service = LeaderService(cfg)
    _MB = 1024 * 1024
    server  = aio.server(options=[
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

    if sys.stdin.isatty():
        # Interactive terminal: show CLI prompt
        try:
            await cli_loop(service)
        finally:
            await server.stop(grace=5)
    else:
        # Non-interactive (subprocess / CI / piped stdin): run until killed.
        # Training starts automatically via --auto-start or when a worker
        # registers and the min_workers threshold is met.
        log.info("Non-interactive mode — Ctrl-C or SIGTERM to stop.")
        try:
            await server.wait_for_termination()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await server.stop(grace=5)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distributed ResNet-101 — Leader")
    p.add_argument("--port",         type=int,   default=50051)
    p.add_argument("--min-workers",  type=int,   default=1,   dest="min_workers",
                   help="Minimum workers before auto-start triggers")
    p.add_argument("--auto-start",   action="store_true",     dest="auto_start",
                   help="Start training automatically when min-workers threshold met")
    p.add_argument("--epochs",       type=int,   default=90)
    p.add_argument("--num-classes",  type=int,   default=200, dest="num_classes")
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--topk",         type=int,   default=50_000,
                   help="Top-K gradient elements per layer (0 = full gradients)")
    p.add_argument("--model",        default="resnet101", dest="model_name",
                   choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--heartbeat-timeout", type=float, default=_DEFAULT_HEARTBEAT_TIMEOUT,
                   dest="heartbeat_timeout",
                   help="Seconds of silence before a worker is marked dead (default 60)")
    p.add_argument("--heartbeat-check-interval", type=float,
                   default=_DEFAULT_HEARTBEAT_CHECK_INTERVAL,
                   dest="heartbeat_check_interval",
                   help="How often to check for dead workers in seconds (default 5)")
    p.add_argument("--grad-sync-timeout", type=float, default=_DEFAULT_GRAD_SYNC_TIMEOUT,
                   dest="grad_sync_timeout",
                   help="Max seconds to wait for all workers per gradient round (default 120)")
    cfg = p.parse_args()
    asyncio.run(main(cfg))
