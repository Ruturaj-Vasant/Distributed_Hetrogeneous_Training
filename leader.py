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

from __future__ import annotations
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

from dataset import make_any_val_loader, get_dataset_info
from proto import trainer_pb2, trainer_pb2_grpc
from run_recorder import RunRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [leader] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("leader")

# ── Tunables ──────────────────────────────────────────────────────────────────

STATUS_INTERVAL     = 10.0      # seconds between auto-printed cluster tables
TINY_IMAGENET_TRAIN = 100_000   # kept for backwards-compat; use _train_samples at runtime
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
    # Queue-based assignment: leader puts a sentinel per training run;
    # worker's GetAssignment call blocks on get() and works across resets.
    assignment_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Populated by _assign_late_worker before queuing RESHARD command;
    # read and cleared inside the Heartbeat handler when building the response.
    pending_reshard: Optional[trainer_pb2.ShardAssignment] = None

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

        # Run recorder (created at training start, closed on reset)
        self._recorder: Optional[RunRecorder] = None

        # Dataset info — resolved from cfg.dataset at training start
        _ds = get_dataset_info(getattr(cfg, "dataset", "tinyimagenet"))
        _full = _ds["train_samples"]
        _cap  = getattr(cfg, "train_samples", None)
        self._train_samples: int = min(_full, _cap) if _cap else _full
        self._num_classes:   int = _ds["num_classes"]

        # Gradient numel — populated by _init_model()
        self._raw_gradient_numel:        int = 0
        self._compressed_gradient_numel: int = 0

        # Workers that registered mid-training but await manual admission
        self._pending_worker_ids: set[str] = set()

        # Gradient synchronisation across workers
        # asyncio.Condition protects _pending_grads and _grad_round.
        # Workers block in wait_for() until the round increments.
        self._grad_cond       = asyncio.Condition()
        self._pending_grads: dict[str, trainer_pb2.GradientPush] = {}
        self._grad_round      = 0
        self._grad_round_start: float = 0.0   # monotonic time when first grad arrived this round
        self._last_payload: bytes = b""

        # Per-worker step latency tracking (set while _grad_cond is held)
        self._grad_arrival: dict[str, float] = {}   # worker_id → monotonic arrival time
        self._prev_round_end: float = 0.0            # when the previous round finished

    # ── RPC: Register ─────────────────────────────────────────────────────────

    async def Register(
        self,
        request: trainer_pb2.RegisterRequest,
        context,
    ) -> trainer_pb2.RegisterResponse:

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

        # All workers always enter the pending pool first.
        # In auto-start mode they are immediately admitted; otherwise the
        # operator must run 'admit' before typing 'start'.
        self._pending_worker_ids.add(wid)

        if self._training_started.is_set():
            log.info(
                f"[pending] {wid} ({hw.hostname}) — training running, "
                f"use 'admit {wid}' in watch.py (triggers Phase-2 reshard)."
            )
        elif self.cfg.auto_start:
            # Auto mode: admit immediately, then check threshold.
            self._pending_worker_ids.discard(wid)
            async with self._lock:
                n_admitted = sum(
                    1 for w in self._workers.values()
                    if w.is_alive and w.worker_id not in self._pending_worker_ids
                )
            log.info(
                f"[auto-admitted] {wid} ({hw.hostname})  "
                f"{n_admitted}/{self.cfg.min_workers} workers ready"
            )
            if n_admitted >= self.cfg.min_workers:
                asyncio.create_task(self.start_training())
        else:
            log.info(
                f"[pending] {wid} ({hw.hostname}) — "
                f"use 'admit {wid}' or 'admit all', then 'start'."
            )

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
                cmd = trainer_pb2.HeartbeatResponse.CONTINUE
                try:
                    cmd = w.cmd_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if cmd == trainer_pb2.HeartbeatResponse.RESHARD and w.pending_reshard:
                    new_shard       = w.pending_reshard
                    w.pending_reshard = None
                    yield trainer_pb2.HeartbeatResponse(command=cmd, new_shard=new_shard)
                else:
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

        # Block until start_training() puts a sentinel in the queue.
        # Using a Queue (not Event) means this works across multiple training
        # runs without needing to replace the primitive on reset.
        await w.assignment_queue.get()

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

        # Gradient push proves the worker is alive — refresh liveness timestamp.
        # Also re-admit a worker that just finished rebuilding its loader after a reshard.
        async with self._lock:
            w = self._workers.get(wid)
            if w is not None:
                w.last_heartbeat = time.monotonic()
                if not w.assigned and w.is_alive:
                    w.assigned = True
                    log.info(
                        f"[reshard-done] {wid} ({w.hostname}) "
                        "rejoined gradient rounds"
                    )

        round_before = self._grad_round

        async with self._grad_cond:
            if not self._pending_grads:
                self._grad_round_start = time.monotonic()   # first grad this round
            self._pending_grads[wid] = request
            self._grad_arrival[wid]  = time.monotonic()
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
            workers     = list(self._workers.values())
            pending_ids = set(self._pending_worker_ids)

        now_wall     = time.time()
        now_mono     = time.monotonic()
        admitted     = [w for w in workers if w.worker_id not in pending_ids]
        total_samples = sum(len(w.shard_indices) for w in admitted) or 1
        summaries = [
            trainer_pb2.WorkerSummary(
                worker_id    = w.worker_id,
                hostname     = w.hostname,
                score        = w.score,
                status       = w.status,
                shard_pct    = len(w.shard_indices) / total_samples * 100,
                last_seen_ts = int(now_wall - (now_mono - w.last_heartbeat)),
            )
            for w in admitted
        ]
        pending = [
            trainer_pb2.PendingWorkerInfo(
                worker_id       = w.worker_id,
                hostname        = w.hostname,
                score           = w.score,
                accel_summary   = w.accel_summary,
                waiting_seconds = int(now_mono - w.registered_at),
            )
            for w in workers
            if w.worker_id in pending_ids
        ]

        return trainer_pb2.ClusterStatusResponse(
            workers          = summaries,
            global_step      = self._global_step,
            global_loss      = self._global_loss,
            current_epoch    = self._current_epoch,
            pending_workers  = pending,
            training_started = self._training_started.is_set(),
        )

    # ── RPC: AdmitWorkers ─────────────────────────────────────────────────────

    async def AdmitWorkers(
        self,
        request: trainer_pb2.AdmitWorkersRequest,
        context,
    ) -> trainer_pb2.AdmitWorkersResponse:
        ids_to_admit = list(request.worker_ids) or list(self._pending_worker_ids)
        admitted, not_found = [], []

        for wid in ids_to_admit:
            if wid not in self._pending_worker_ids:
                not_found.append(wid)
                continue
            async with self._lock:
                w = self._workers.get(wid)
            if w is None or not w.is_alive:
                self._pending_worker_ids.discard(wid)
                not_found.append(wid)
                continue

            if self._training_started.is_set():
                # Mid-training: Phase-2 full shard rebalancing.
                await self._assign_late_worker(w)
                log.info(f"[admit] {wid} ({w.hostname}) admitted — Phase-2 reshard triggered.")
            else:
                # Pre-training: just move out of the pending pool.
                # The worker will get its shard when 'start' is typed.
                log.info(
                    f"[admit] {wid} ({w.hostname}) admitted — "
                    "will join at next 'start'."
                )

            self._pending_worker_ids.discard(wid)
            admitted.append(wid)

        return trainer_pb2.AdmitWorkersResponse(
            admitted_count = len(admitted),
            admitted_ids   = admitted,
            not_found_ids  = not_found,
        )

    # ── Public: kick off training ─────────────────────────────────────────────

    async def start_training(self) -> None:
        if self._training_started.is_set():
            log.warning("Training is already running.")
            return

        async with self._lock:
            # Prefer explicitly admitted workers (not in pending pool).
            alive = [
                w for w in self._workers.values()
                if w.is_alive and w.worker_id not in self._pending_worker_ids
            ]
            if not alive:
                # Nobody was explicitly admitted — fall back to admitting everyone.
                # This preserves the simple workflow: workers connect, type 'start',
                # training begins.  The gate only matters when you want a subset.
                log.info(
                    "No workers explicitly admitted — admitting all pending workers."
                )
                alive = [w for w in self._workers.values() if w.is_alive]
                for w in alive:
                    self._pending_worker_ids.discard(w.worker_id)

        if not alive:
            log.error("No alive workers — cannot start.")
            return

        # Score-proportional shard assignment
        indices = list(range(self._train_samples))
        log.info(f"Computing shards for {len(alive)} worker(s) …")
        self._compute_shards(alive, indices)

        # Initialise model and optimizer on leader (in thread — CPU-bound)
        log.info("Initialising ResNet-101 on leader …")
        await asyncio.get_event_loop().run_in_executor(None, self._init_model)

        self._training_cfg = trainer_pb2.TrainingConfig(
            model_name       = getattr(self.cfg, "model_name", "resnet101"),
            dataset          = getattr(self.cfg, "dataset", "tinyimagenet"),
            num_classes      = self._num_classes,
            total_epochs     = self.cfg.epochs,
            base_lr          = self.cfg.lr,
            weight_decay     = self.cfg.weight_decay,
            gradient_topk_k  = self.cfg.topk,
            grad_accum_steps = getattr(self.cfg, "grad_accum", 1),
        )

        self._training_started.set()

        # Create run recorder
        topk_k     = getattr(self.cfg, "topk", 0)
        world_size = len(alive)
        self._recorder = RunRecorder(
            model_name                = getattr(self.cfg, "model_name", "resnet101"),
            dataset                   = "tinyimagenet200",
            optimizer                 = "sgd",
            topk_k                    = topk_k,
            epochs                    = self.cfg.epochs,
            lr                        = self.cfg.lr,
            weight_decay              = self.cfg.weight_decay,
            batch_size                = getattr(self.cfg, "batch_size", 32),
            world_size                = world_size,
            dataset_samples           = self._train_samples,
            raw_gradient_numel        = self._raw_gradient_numel,
            compressed_gradient_numel = self._compressed_gradient_numel,
            runs_root                 = getattr(self.cfg, "runs_root", "runs"),
        )
        log.info(f"Run recorder: {self._recorder.run_dir}")

        # Unblock every waiting worker's GetAssignment call
        for w in alive:
            w.assignment_queue.put_nowait(True)

        log.info(
            f"Training started — assignments dispatched to {len(alive)} worker(s)."
        )

    # ── Public: reset for next training run ──────────────────────────────────

    async def reset_training(self) -> None:
        """
        Reset leader state so the operator can run another training job
        with the same connected workers.  Workers will automatically loop
        back to GetAssignment and block until the next 'start'.
        """
        if not self._training_started.is_set():
            log.warning("Nothing to reset — training has not started.")
            return

        self._training_started = asyncio.Event()   # fresh, unset

        if self._recorder is not None:
            run_dir = self._recorder.close()
            log.info(f"Run saved to {run_dir}")
            self._recorder = None

        async with self._grad_cond:
            self._pending_grads.clear()
            self._grad_round      = 0
            self._grad_round_start = 0.0

        self._global_step   = 0
        self._current_epoch = 0
        self._global_loss   = 0.0
        self._last_payload  = b""
        self._prev_state    = {}
        self._model         = None
        self._optimizer     = None

        async with self._lock:
            n = len(self._workers)

        log.info(
            f"Reset complete — {n} worker(s) still connected and waiting. "
            "Type 'start' to begin the next run."
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
                    self._pending_worker_ids.discard(w.worker_id)

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
            step    = self._global_step
            epoch   = self._current_epoch
            loss    = self._global_loss

        if not workers:
            return

        print(
            f"  Step={step:>6}  "
            f"Epoch={epoch:>3}  "
            f"Loss={loss:.4f}  "
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
        # Only count workers that have completed GetAssignment (assigned=True).
        # Late joiners that are still loading data must not block the grad round.
        async with self._lock:
            return sum(1 for w in self._workers.values() if w.is_alive and w.assigned)

    async def _assign_late_worker(self, w: WorkerState) -> None:
        """
        Phase 2 shard rebalancing: redistribute the full dataset proportionally
        across all active workers + the newly admitted worker.

        Existing workers are sent a RESHARD command (via the next heartbeat) and
        temporarily removed from gradient rounds while they reload their data.
        They re-join automatically when their first new ExchangeGradients call arrives.
        """
        async with self._lock:
            active = [
                wr for wr in self._workers.values()
                if wr.is_alive and (wr.assigned or wr.worker_id == w.worker_id)
            ]

        if w not in active:
            active.append(w)

        # Score-proportional reshard across the full dataset
        indices = list(range(self._train_samples))
        self._compute_shards(active, indices)

        # Send RESHARD to each existing worker and exclude it from grad rounds
        # until it signals it has rebuilt its loader (first ExchangeGradients call).
        async with self._lock:
            for wr in active:
                if wr.worker_id == w.worker_id:
                    continue
                wr.pending_reshard = trainer_pb2.ShardAssignment(
                    worker_id        = wr.worker_id,
                    indices          = wr.shard_indices,
                    local_batch_size = wr.local_batch_size,
                    primary_device   = _infer_device(wr),
                )
                wr.assigned = False  # temporarily excluded from grad rounds
                wr.cmd_queue.put_nowait(trainer_pb2.HeartbeatResponse.RESHARD)
                log.info(
                    f"  reshard cmd → {wr.hostname:<22}  "
                    f"samples={len(wr.shard_indices):,}  batch={wr.local_batch_size}"
                )

        # Unblock the new worker's GetAssignment — it picks up its shard directly.
        w.assignment_queue.put_nowait(True)
        log.info(
            f"  new shard   → {w.hostname:<22}  "
            f"samples={len(w.shard_indices):,}  batch={w.local_batch_size}"
        )

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
            base = getattr(self.cfg, "batch_size", 32)
            w.local_batch_size = max(8, min(256, int(base * frac * len(workers))))
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
            weights=None, num_classes=self._num_classes
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

        # Gradient element counts for compression tracking
        self._raw_gradient_numel = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        topk_k = getattr(self.cfg, "topk", 0)
        if topk_k > 0:
            num_grad_layers = sum(
                1 for p in self._model.parameters() if p.requires_grad
            )
            self._compressed_gradient_numel = topk_k * num_grad_layers
        else:
            self._compressed_gradient_numel = self._raw_gradient_numel

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
                shape   = list(sg.shape)
                n_elems = 1
                for d in shape:
                    n_elems *= d
                # Build sparse grad on CPU (fast list→tensor), then one device transfer.
                # Creating tensors from Python lists directly on MPS/CUDA is slow
                # because PyTorch materialises a CPU tensor first anyway.
                indices_t = torch.tensor(sg.indices, dtype=torch.long)
                values_t  = torch.tensor(sg.values,  dtype=torch.float32)
                flat_cpu  = torch.zeros(n_elems)
                flat_cpu.scatter_(0, indices_t, values_t)
                grad = flat_cpu.reshape(shape).to(self._device)
                if param.grad is None:
                    param.grad = grad * weight
                else:
                    param.grad.add_(grad, alpha=weight)

        self._optimizer.step()
        old_epoch = self._current_epoch
        self._global_step += 1

        losses = [g.loss for g in grads.values() if g.loss > 0]
        if losses:
            self._global_loss = sum(losses) / len(losses)

        # Compute parameter delta (new - old) and serialise as float16
        # float32 → float16 halves transfer size (43 MB → 22 MB) with negligible
        # precision loss for typical LR-scaled deltas (magnitude << 1.0).
        # apply_delta() on the worker uses add_() which handles mixed precision.
        new_state  = self._model.state_dict()
        delta = {
            k: (v.cpu() - self._prev_state[k]).half()
            for k, v in new_state.items()
            if k in self._prev_state
        }
        self._prev_state = {k: v.cpu().clone() for k, v in new_state.items()}

        buf = io.BytesIO()
        torch.save(delta, buf)
        self._last_payload = buf.getvalue()
        # ── Per-worker step latency ───────────────────────────────────────────
        round_end     = time.monotonic()
        arrivals      = dict(self._grad_arrival)
        prev_end      = self._prev_round_end
        self._prev_round_end = round_end
        self._pending_grads.clear()
        self._grad_arrival.clear()

        # Build per-worker stats: compute time (train step) + wait time (straggler idle)
        latency_parts = []
        max_wait_ms   = 0.0
        for wid, arrival in arrivals.items():
            compute_ms = (arrival - prev_end) * 1000 if prev_end > 0 else 0.0
            wait_ms    = (round_end - arrival) * 1000
            max_wait_ms = max(max_wait_ms, wait_ms)
            hostname   = self._workers[wid].hostname if wid in self._workers else wid
            latency_parts.append((hostname, compute_ms, wait_ms))

        round_ms          = (round_end - min(arrivals.values())) * 1000 if arrivals else 0.0
        straggler_delay_s = (max_wait_ms / 1000.0) if len(arrivals) > 1 else 0.0

        if self._global_step % 10 == 0 or len(arrivals) > 1:
            parts_str = "  ".join(
                f"{h}: compute={c:.0f}ms wait={w:.0f}ms"
                + (" ← straggler" if w == max_wait_ms and w > 200 and len(arrivals) > 1 else "")
                for h, c, w in latency_parts
            )
            log.info(
                f"Step {self._global_step:>6,}  loss={self._global_loss:.4f}  "
                f"round={round_ms:.0f}ms  delta={len(self._last_payload) // 1024:,}KB"
                + (f"\n          {parts_str}" if len(arrivals) > 1 else "")
            )

        # Epoch tracking — accounts for grad accumulation reducing sync count
        grad_accum      = getattr(self.cfg, "grad_accum", 1)
        batch_size      = getattr(self.cfg, "batch_size", 32)
        steps_per_epoch = max(
            1,
            self._train_samples // (batch_size * max(1, len(grads)) * grad_accum),
        )
        self._current_epoch = self._global_step // steps_per_epoch

        if self._recorder:
            self._recorder.log_step(
                self._global_step, self._global_loss, round_ms, straggler_delay_s
            )

        if self._current_epoch > old_epoch:
            val_acc = self._run_val_sync()
            if self._recorder:
                self._recorder.log_epoch(self._current_epoch, val_acc)

    # ── Private: validation evaluation (runs in thread pool) ─────────────────

    def _run_val_sync(self) -> float:
        """Top-1 validation accuracy."""
        data_root = getattr(self.cfg, "data_root", None)
        dataset   = getattr(self.cfg, "dataset", "tinyimagenet")
        try:
            loader = make_any_val_loader(dataset, data_root, batch_size=256, cpu_cores=4)
        except Exception as exc:
            log.warning(f"Val loader unavailable: {exc}")
            return 0.0

        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                preds  = self._model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        self._model.train()

        acc = correct / total if total > 0 else 0.0
        log.info(
            f"Epoch {self._current_epoch} — val_acc={acc:.4f}  "
            f"({correct}/{total})"
        )
        return acc

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
            sys.exit(0)
        else:
            print(f"  Unknown command: {cmd!r}")
            print("  Try: admit [id ...] | admit all | start | status | reset | quit")


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
            if service._recorder is not None:
                run_dir = service._recorder.close()
                log.info(f"Run saved to {run_dir}")
            await server.stop(grace=5)
    else:
        # Non-interactive (subprocess / CI / piped stdin): run until killed.
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distributed ResNet-101 — Leader")
    p.add_argument("--port",         type=int,   default=50051)
    p.add_argument("--min-workers",  type=int,   default=1,   dest="min_workers",
                   help="Minimum workers before auto-start triggers")
    p.add_argument("--auto-start",   action="store_true",     dest="auto_start",
                   help="Start training automatically when min-workers threshold met")
    p.add_argument("--dataset",      default="tinyimagenet",
                   choices=["tinyimagenet", "cifar10"],
                   help="Dataset to train on (default: tinyimagenet)")
    p.add_argument("--epochs",       type=int,   default=90)
    p.add_argument("--train-samples", type=int,  default=None, dest="train_samples",
                   help="Cap the number of training samples per epoch (default: use full dataset)")
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--batch-size",   type=int,   default=32,   dest="batch_size",
                   help="Base batch size per worker (scales proportionally with score, default 32)")
    p.add_argument("--topk",         type=int,   default=50_000,
                   help="Top-K gradient elements per layer (0 = full gradients)")
    p.add_argument("--grad-accum",   type=int,   default=1, dest="grad_accum",
                   help="Gradient accumulation steps — sync every N batches (default 1)")
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
    p.add_argument("--data-root", default=None, dest="data_root",
                   help="Path to Tiny ImageNet-200 cache (default: ~/.cache/tiny-imagenet-200)")
    cfg = p.parse_args()
    asyncio.run(main(cfg))
