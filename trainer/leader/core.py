"""
trainer/leader/core.py — LeaderCore mixin.

Contains all computation methods (model management, gradient aggregation,
shard scheduling, validation) as a mixin so that LeaderService inherits
them without any changes to self.* access patterns.

Assumed self.* attributes (set by LeaderService.__init__):
    cfg, _device, _model, _optimizer, _prev_state
    _workers, _lock
    _pending_grads, _grad_arrival, _prev_round_end
    _global_step, _current_epoch, _global_loss, _last_payload
    _raw_gradient_numel, _compressed_gradient_numel
    _recorder, _train_samples, _num_classes
"""
from __future__ import annotations
import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision.models as models

from dataset import make_any_val_loader
from proto import trainer_pb2
from trainer.leader.worker_state import WorkerState, infer_device
from trainer.core.logging import get as _get_log

log = _get_log("leader")


class LeaderCore:
    """Computation mixin for LeaderService — never instantiated directly."""

    # ── Device selection ──────────────────────────────────────────────────────

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

    # ── Alive-worker count ────────────────────────────────────────────────────

    async def _count_alive(self) -> int:
        """Only count workers that completed GetAssignment (assigned=True)."""
        async with self._lock:
            return sum(1 for w in self._workers.values() if w.is_alive and w.assigned)

    # ── Model initialisation ──────────────────────────────────────────────────

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

    # ── Gradient aggregation + optimizer step ─────────────────────────────────

    def _aggregate_and_step(self) -> None:
        """
        Weighted gradient aggregation (runs in thread pool, safe because
        _grad_cond is held by the caller — no concurrent ExchangeGradients
        can enter while we run).

        Worker weight = local_batch_count / total_samples — equivalent to
        computing the gradient over the full union of batches as one model.
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
                # Build sparse grad on CPU first, then one device transfer.
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

        # Compute weight delta (float16 halves transfer: 43 MB → 22 MB)
        new_state = self._model.state_dict()
        delta = {
            k: (v.cpu() - self._prev_state[k]).half()
            for k, v in new_state.items()
            if k in self._prev_state
        }
        self._prev_state = {k: v.cpu().clone() for k, v in new_state.items()}

        buf = io.BytesIO()
        torch.save(delta, buf)
        self._last_payload = buf.getvalue()

        # Per-worker step latency
        round_end   = time.monotonic()
        arrivals    = dict(self._grad_arrival)
        prev_end    = self._prev_round_end
        self._prev_round_end = round_end
        self._pending_grads.clear()
        self._grad_arrival.clear()

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

    # ── Validation ────────────────────────────────────────────────────────────

    def _run_val_sync(self) -> float:
        """Top-1 validation accuracy (runs in thread pool)."""
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
            f"Epoch {self._current_epoch} — val_acc={acc:.4f}  ({correct}/{total})"
        )
        return acc

    # ── Model serialisation ───────────────────────────────────────────────────

    def _serialize_full_model(self) -> bytes:
        buf = io.BytesIO()
        torch.save(
            {k: v.cpu() for k, v in self._model.state_dict().items()},
            buf,
        )
        return buf.getvalue()

    # ── Shard distribution ────────────────────────────────────────────────────

    def _compute_shards(
        self, workers: list[WorkerState], indices: list[int]
    ) -> None:
        total_score = sum(w.score for w in workers) or 1.0
        offset = 0
        for i, w in enumerate(workers):
            frac = w.score / total_score
            n = (
                len(indices) - offset
                if i == len(workers) - 1
                else int(frac * len(indices))
            )
            w.shard_indices    = indices[offset: offset + n]
            base = getattr(self.cfg, "batch_size", 32)
            w.local_batch_size = max(8, min(256, int(base * frac * len(workers))))
            offset += n
            log.info(
                f"  {w.hostname:<22}  score={w.score:>8.1f}  "
                f"frac={frac * 100:>5.1f}%  samples={n:>6,}  "
                f"batch={w.local_batch_size}"
            )

    async def _assign_late_worker(self, w: WorkerState) -> None:
        """
        Phase-2 reshard: redistribute full dataset across all active + new worker.
        Existing workers get RESHARD command and are temporarily excluded from
        gradient rounds until their first new ExchangeGradients call.
        """
        async with self._lock:
            active = [
                wr for wr in self._workers.values()
                if wr.is_alive and (wr.assigned or wr.worker_id == w.worker_id)
            ]

        if w not in active:
            active.append(w)

        indices = list(range(self._train_samples))
        self._compute_shards(active, indices)

        async with self._lock:
            for wr in active:
                if wr.worker_id == w.worker_id:
                    continue
                wr.pending_reshard = trainer_pb2.ShardAssignment(
                    worker_id        = wr.worker_id,
                    indices          = wr.shard_indices,
                    local_batch_size = wr.local_batch_size,
                    primary_device   = infer_device(wr),
                )
                wr.assigned = False
                wr.cmd_queue.put_nowait(trainer_pb2.HeartbeatResponse.RESHARD)
                log.info(
                    f"  reshard cmd → {wr.hostname:<22}  "
                    f"samples={len(wr.shard_indices):,}  batch={wr.local_batch_size}"
                )

        w.assignment_queue.put_nowait(True)
        log.info(
            f"  new shard   → {w.hostname:<22}  "
            f"samples={len(w.shard_indices):,}  batch={w.local_batch_size}"
        )
