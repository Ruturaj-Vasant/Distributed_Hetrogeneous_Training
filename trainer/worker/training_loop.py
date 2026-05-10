"""
trainer/worker/training_loop.py — Synchronous inner training loop.
Runs in a thread pool so asyncio stays free for heartbeat + RPC.
"""
from __future__ import annotations
import time

import torch
import torch.nn as nn
import torch.utils.data
import grpc

from proto import trainer_pb2, trainer_pb2_grpc
from trainer.worker.gradients import compress_gradients, apply_delta
from trainer.worker.heartbeat import WorkerSharedState
from trainer.core.logging import get as _get_log
from trainer.core.errors import fmt_grpc_error

log = _get_log("worker")


def training_loop(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    sync_stub:   trainer_pb2_grpc.TrainerServiceStub,
    worker_id:   str,
    config:      trainer_pb2.TrainingConfig,
    device:      torch.device,
    shared:      WorkerSharedState,
    start_epoch: int = 0,
) -> None:
    """
    Each step:
      1. Forward + loss.backward()      (gradients on device)
      2. Top-K compress gradients        (move to CPU)
      3. ExchangeGradients RPC           (blocks until all workers sync)
      4. Apply weight delta from leader  (add delta to local params)
      5. Repeat
    """
    criterion = nn.CrossEntropyLoss()
    topk_k    = config.gradient_topk_k
    epochs    = config.total_epochs
    grad_accum = getattr(config, "grad_accum_steps", 1) or 1

    log.info(
        f"Training  epochs={epochs}  "
        f"batches/epoch≈{len(loader)}  "
        f"topk_k={topk_k}  grad_accum={grad_accum}  device={device}"
    )

    for epoch in range(start_epoch, epochs):
        shared.epoch_completed = epoch
        shared.status = trainer_pb2.TRAINING
        epoch_losses: list[float] = []

        accum_loss    = 0.0
        accum_samples = 0
        accum_count   = 0
        model.zero_grad()

        for batch_idx, (images, labels) in enumerate(loader):
            if shared.stop:
                log.info("Stop flag set — exiting training loop.")
                return

            while shared.paused and not shared.stop:
                time.sleep(0.5)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            (loss / grad_accum).backward()

            loss_val    = loss.item()
            shared.loss = loss_val
            epoch_losses.append(loss_val)
            accum_loss    += loss_val
            accum_samples += images.size(0)
            accum_count   += 1

            is_last_batch = (batch_idx + 1) == len(loader)
            if accum_count < grad_accum and not is_last_batch:
                if batch_idx % 20 == 0:
                    log.info(
                        f"E{epoch + 1:02d}/{epochs}  "
                        f"B{batch_idx:04d}/{len(loader)}  "
                        f"loss={loss_val:.4f}  "
                        f"step={shared.steps}  (accum {accum_count}/{grad_accum})"
                    )
                continue

            sparse_grads = compress_gradients(model, topk_k)
            shared.steps += 1
            avg_loss_accum = accum_loss / accum_count

            try:
                update = sync_stub.ExchangeGradients(
                    trainer_pb2.GradientPush(
                        worker_id         = worker_id,
                        global_step       = shared.steps,
                        local_batch_count = accum_samples,
                        loss              = avg_loss_accum,
                        gradients         = sparse_grads,
                    )
                )
            except grpc.RpcError as exc:
                log.error(f"ExchangeGradients failed: {fmt_grpc_error(exc)}")
                shared.leader_disconnected = True
                shared.stop = True
                return

            apply_delta(model, update.payload, device)
            model.zero_grad()

            if batch_idx % 20 == 0:
                log.info(
                    f"E{epoch + 1:02d}/{epochs}  "
                    f"B{batch_idx:04d}/{len(loader)}  "
                    f"loss={avg_loss_accum:.4f}  "
                    f"step={shared.steps}"
                )

            accum_loss    = 0.0
            accum_samples = 0
            accum_count   = 0

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        shared.epoch_completed = epoch + 1
        log.info(f"Epoch {epoch + 1}/{epochs} done  avg_loss={avg_loss:.4f}")

    shared.status = trainer_pb2.IDLE
    log.info("Training complete.")
