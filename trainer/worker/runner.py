"""
trainer/worker/runner.py — Outer worker coroutine: reconnect loop + job loop.
Orchestrates all worker subsystems; called by the thin worker.py entry point.
"""
from __future__ import annotations
import asyncio
import logging

import grpc
import torch
import torchvision.models as models

from proto import trainer_pb2, trainer_pb2_grpc
from dataset import ensure_any_dataset, make_any_train_loader, get_dataset_info
from hardware_probe import probe

from trainer.worker.connection import connect_and_register
from trainer.worker.heartbeat import WorkerSharedState, heartbeat_loop
from trainer.worker.training_loop import training_loop
from trainer.worker.synthetic_data import make_synthetic_loader
from trainer.worker.gradients import load_full_weights
from trainer.worker.proto_helpers import parse_dataset_list, resolve_device
from trainer.core.logging import get as _get_log
from trainer.core.errors import LeaderConnectionError, RegistrationRejectedError

log = _get_log("worker")

_MODEL_FNS = {
    "resnet18":  models.resnet18,
    "resnet50":  models.resnet50,
    "resnet101": models.resnet101,
}


async def run(cfg) -> None:
    # ── Step 1: hardware probe ────────────────────────────────────────────────
    log.info("Probing hardware …")
    hw_result = probe()
    hw        = hw_result.hw
    bm        = hw_result.benchmark
    log.info(
        f"Score={bm.score:.1f}  "
        f"Bench={bm.forward_ms:.1f}ms  "
        f"FreeMem={bm.memory_free_gb:.1f}GB"
    )

    # ── Step 2: dataset download ──────────────────────────────────────────────
    preload_datasets = parse_dataset_list(cfg.preload_datasets, cfg.dataset)
    if cfg.dry_run:
        log.info("Dry-run mode: skipping dataset download (synthetic data will be used)")
    else:
        log.info(f"Ensuring {cfg.dataset} dataset is present …")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: ensure_any_dataset(cfg.dataset, cfg.cache_dir)
        )
        for dataset_name in preload_datasets:
            if dataset_name == cfg.dataset:
                continue
            log.info(f"Ensuring {dataset_name} dataset is present …")
            await asyncio.get_event_loop().run_in_executor(
                None, lambda name=dataset_name: ensure_any_dataset(name, cfg.cache_dir)
            )

    # ── Reconnect loop: re-registers after leader disconnection ───────────────
    while True:

        try:
            async_ch, async_stub, sync_ch, sync_stub, reg = \
                await connect_and_register(cfg, hw, bm)
        except LeaderConnectionError as exc:
            log.error(str(exc))
            raise

        if not reg.accepted:
            log.error(f"Registration rejected: {reg.reject_reason}")
            await async_ch.close()
            sync_ch.close()
            raise RegistrationRejectedError(reg.reject_reason)

        worker_id = reg.worker_id
        log.info(f"Registered as worker_id={worker_id}")

        shared  = WorkerSharedState()
        hb_task = asyncio.create_task(
            heartbeat_loop(async_stub, worker_id, shared)
        )

        try:
            # Job loop — repeats for each training run on this leader
            while True:
                shared.status             = trainer_pb2.IDLE
                shared.stop               = False
                shared.leader_disconnected = False
                shared.loss               = -1.0
                shared.steps              = 0
                log.info("Waiting for assignment from leader …  (leader must type 'start')")

                try:
                    assignment_resp = await async_stub.GetAssignment(
                        trainer_pb2.GetAssignmentRequest(worker_id=worker_id)
                    )
                except grpc.RpcError as exc:
                    log.warning(
                        f"GetAssignment lost ({exc.code().name}) — "
                        "leader may have restarted."
                    )
                    shared.leader_disconnected = True
                    break

                assignment = assignment_resp.assignment
                config     = assignment_resp.config
                device     = resolve_device(assignment.primary_device)

                log.info(
                    f"Assignment received:  "
                    f"{len(assignment.indices):,} samples  "
                    f"batch={assignment.local_batch_size}  "
                    f"device={device}"
                )

                model_fn = _MODEL_FNS.get(config.model_name, models.resnet101)
                log.info(
                    f"Building {config.model_name} ({config.num_classes} classes) "
                    f"on {device} …"
                )
                model = model_fn(weights=None, num_classes=config.num_classes)
                load_full_weights(model, assignment_resp.model_weights, device)
                model = model.to(device).train()

                dataset_name = config.dataset or "tinyimagenet"
                if cfg.dry_run:
                    n_synthetic = min(len(assignment.indices), 320)
                    loader = make_synthetic_loader(
                        n_synthetic, assignment.local_batch_size, config.num_classes
                    )
                    log.info(
                        f"Synthetic DataLoader: {n_synthetic} samples  "
                        f"batch={assignment.local_batch_size}"
                    )
                else:
                    if dataset_name not in preload_datasets:
                        log.info(
                            f"Leader requested {dataset_name!r} — downloading now …"
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None, lambda: ensure_any_dataset(dataset_name, cfg.cache_dir)
                        )
                    loader = make_any_train_loader(
                        dataset    = dataset_name,
                        root       = cfg.cache_dir,
                        indices    = list(assignment.indices),
                        batch_size = assignment.local_batch_size,
                        cpu_cores  = hw.cpu_cores,
                    )
                log.info(
                    f"DataLoader ready:  {len(loader.dataset):,} samples  "
                    f"batch={assignment.local_batch_size}  "
                    f"batches/epoch={len(loader)}"
                )

                # Training + reshard loop
                while True:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: training_loop(
                            model, loader, sync_stub,
                            worker_id, config, device, shared,
                            start_epoch=shared.epoch_completed,
                        ),
                    )

                    if shared.reshard_indices is not None:
                        new_indices    = shared.reshard_indices
                        new_batch_size = shared.reshard_batch_size
                        shared.reshard_indices    = None
                        shared.reshard_batch_size = 0
                        shared.stop               = False
                        log.info(
                            f"Resharding: {len(new_indices):,} new samples  "
                            f"batch={new_batch_size}  "
                            f"continuing from epoch {shared.epoch_completed}"
                        )
                        if cfg.dry_run:
                            n_synthetic = min(len(new_indices), 320)
                            loader = make_synthetic_loader(
                                n_synthetic, new_batch_size, config.num_classes
                            )
                        else:
                            loader = make_any_train_loader(
                                dataset    = dataset_name,
                                root       = cfg.cache_dir,
                                indices    = new_indices,
                                batch_size = new_batch_size,
                                cpu_cores  = hw.cpu_cores,
                            )
                        continue

                    break

                if shared.stop:
                    if shared.leader_disconnected:
                        log.warning("Leader disconnected during training.")
                        break
                    else:
                        log.info("Stopping on leader instruction.")
                        return

                log.info(
                    "Training complete. Staying connected — "
                    "waiting for next job (leader: 'reset' then 'start')."
                )

        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            await async_ch.close()
            sync_ch.close()

        if not shared.leader_disconnected:
            log.info("Worker shut down.")
            return

        log.info(
            "Leader disconnected. Waiting 10 s before reconnecting … "
            "(start a new leader and this worker will re-register automatically)"
        )
        await asyncio.sleep(10)
