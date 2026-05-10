"""
trainer/leader/service.py — LeaderService gRPC servicer.

Contains only the gRPC RPC handlers and training lifecycle methods.
All computation (model, aggregation, sharding) lives in LeaderCore.
"""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Optional

import torch.nn as nn
import grpc

from proto import trainer_pb2, trainer_pb2_grpc
from dataset import get_dataset_info
from run_recorder import RunRecorder
from trainer.leader.core import LeaderCore
from trainer.leader.worker_state import WorkerState, infer_device
from trainer.core.logging import get as _get_log
from trainer.core.errors import GradientSyncError

log = _get_log("leader")

_DEFAULT_HEARTBEAT_TIMEOUT        = 60.0
_DEFAULT_HEARTBEAT_CHECK_INTERVAL =  5.0
_DEFAULT_GRAD_SYNC_TIMEOUT        = 120.0
STATUS_INTERVAL                   = 10.0


class LeaderService(LeaderCore, trainer_pb2_grpc.TrainerServiceServicer):

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()

        self._training_started = asyncio.Event()
        self._training_cfg: Optional[trainer_pb2.TrainingConfig] = None

        self._device     = self._pick_device()
        self._model:     Optional[nn.Module]             = None
        self._optimizer  = None
        self._prev_state: dict = {}

        self._global_step   = 0
        self._current_epoch = 0
        self._global_loss   = 0.0

        self._recorder: Optional[RunRecorder] = None

        _ds = get_dataset_info(getattr(cfg, "dataset", "tinyimagenet"))
        _full = _ds["train_samples"]
        _cap  = getattr(cfg, "train_samples", None)
        self._train_samples: int = min(_full, _cap) if _cap else _full
        self._num_classes:   int = _ds["num_classes"]

        self._raw_gradient_numel:        int = 0
        self._compressed_gradient_numel: int = 0

        self._pending_worker_ids: set[str] = set()

        self._grad_cond       = asyncio.Condition()
        self._pending_grads: dict[str, trainer_pb2.GradientPush] = {}
        self._grad_round      = 0
        self._grad_round_start: float = 0.0
        self._last_payload: bytes = b""

        self._grad_arrival:   dict[str, float] = {}
        self._prev_round_end: float = 0.0

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

        self._pending_worker_ids.add(wid)

        if self._training_started.is_set():
            log.info(
                f"[pending] {wid} ({hw.hostname}) — training running, "
                f"use 'admit {wid}' to trigger Phase-2 reshard."
            )
        elif self.cfg.auto_start:
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

    # ── RPC: Heartbeat ────────────────────────────────────────────────────────

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

    # ── RPC: GetAssignment ────────────────────────────────────────────────────

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
        await w.assignment_queue.get()

        async with self._lock:
            w = self._workers.get(wid)
        if w is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Worker {wid} was removed")
            return

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
                primary_device   = infer_device(w),
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
        wid = request.worker_id

        async with self._lock:
            w = self._workers.get(wid)
            if w is not None:
                w.last_heartbeat = time.monotonic()
                if not w.assigned and w.is_alive:
                    w.assigned = True
                    log.info(
                        f"[reshard-done] {wid} ({w.hostname}) rejoined gradient rounds"
                    )

        round_before = self._grad_round

        async with self._grad_cond:
            if not self._pending_grads:
                self._grad_round_start = time.monotonic()
            self._pending_grads[wid] = request
            self._grad_arrival[wid]  = time.monotonic()
            n_alive    = await self._count_alive()
            n_received = len(self._pending_grads)

            log.debug(
                f"Grad push from {wid}  step={request.global_step}  "
                f"{n_received}/{n_alive} received"
            )

            if n_received >= n_alive:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._aggregate_and_step
                )
                self._grad_round += 1
                self._grad_cond.notify_all()
            else:
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
                await self._assign_late_worker(w)
                log.info(f"[admit] {wid} ({w.hostname}) admitted — Phase-2 reshard triggered.")
            else:
                log.info(
                    f"[admit] {wid} ({w.hostname}) admitted — will join at next 'start'."
                )

            self._pending_worker_ids.discard(wid)
            admitted.append(wid)

        return trainer_pb2.AdmitWorkersResponse(
            admitted_count = len(admitted),
            admitted_ids   = admitted,
            not_found_ids  = not_found,
        )

    # ── Training lifecycle ────────────────────────────────────────────────────

    async def start_training(self) -> None:
        if self._training_started.is_set():
            log.warning("Training is already running.")
            return

        async with self._lock:
            alive = [
                w for w in self._workers.values()
                if w.is_alive and w.worker_id not in self._pending_worker_ids
            ]
            if not alive:
                log.info("No workers explicitly admitted — admitting all pending workers.")
                alive = [w for w in self._workers.values() if w.is_alive]
                for w in alive:
                    self._pending_worker_ids.discard(w.worker_id)

        if not alive:
            log.error("No alive workers — cannot start.")
            return

        indices = list(range(self._train_samples))
        log.info(f"Computing shards for {len(alive)} worker(s) …")
        self._compute_shards(alive, indices)

        log.info(f"Initialising {getattr(self.cfg, 'model_name', 'resnet101')} on leader …")
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

        for w in alive:
            w.assignment_queue.put_nowait(True)

        log.info(f"Training started — assignments dispatched to {len(alive)} worker(s).")

    async def reset_training(self) -> None:
        if not self._training_started.is_set():
            log.warning("Nothing to reset — training has not started.")
            return

        self._training_started = asyncio.Event()

        if self._recorder is not None:
            run_dir = self._recorder.close()
            log.info(f"Run saved to {run_dir}")
            self._recorder = None

        async with self._grad_cond:
            self._pending_grads.clear()
            self._grad_round       = 0
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
            f"Reset complete — {n} worker(s) still connected. "
            "Type 'start' to begin the next run."
        )

    # ── Background tasks ──────────────────────────────────────────────────────

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

                    if dead and 0 < n_received >= n_alive:
                        should_force = True
                        reason = f"{len(dead)} worker(s) removed ({n_received}/{n_alive} grads)"
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
            f"  Step={step:>6}  Epoch={epoch:>3}  "
            f"Loss={loss:.4f}  Workers={len(workers)}"
        )
