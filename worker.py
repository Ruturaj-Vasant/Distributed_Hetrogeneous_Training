"""
worker.py  —  Distributed training worker

Usage:
    python3 worker.py --leader leader-macbook-pro.taila5426e.ts.net [--port 50051]

What this script does, in order:
    1.  Probe local hardware (CPU, CUDA/MPS, RAM) and compute a capability score
    2.  Download Tiny ImageNet-200 to ~/.cache/tiny-imagenet-200 if not present
    3.  Connect to the leader over Tailscale and register
    4.  Start a bidirectional heartbeat stream in the background
    5.  Call GetAssignment — BLOCKS until the operator types "start" on the leader
    6.  Receive shard indices, TrainingConfig, and initial model weights
    7.  Run the training loop:
            forward → loss.backward() → Top-K compress → ExchangeGradients RPC
            → apply weight delta from leader → repeat
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import grpc
from grpc import aio

from proto import trainer_pb2, trainer_pb2_grpc
from hardware_probe import probe, AccelType
from dataset import ensure_dataset, make_train_loader, IMG_SIZE, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")


# ── Synthetic dataloader (dry-run / testing) ──────────────────────────────────

class _SyntheticDataset(torch.utils.data.Dataset):
    """Random tensors that look like Tiny ImageNet batches — no disk I/O."""
    def __init__(self, n: int, num_classes: int) -> None:
        self.n = n
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            torch.randn(3, IMG_SIZE, IMG_SIZE),
            torch.randint(0, self.num_classes, (1,)).item(),
        )


def _make_synthetic_loader(n: int, batch_size: int, num_classes: int) -> torch.utils.data.DataLoader:
    ds = _SyntheticDataset(n, num_classes)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )


# ── gRPC channel helpers ──────────────────────────────────────────────────────

# ResNet-101 state dict ≈ 170 MB; allow 256 MB messages on both sides.
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length",    256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]


def _sync_channel(host: str, port: int) -> grpc.Channel:
    """Synchronous channel — used in the training thread."""
    return grpc.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)


def _async_channel(host: str, port: int) -> aio.Channel:
    """Async channel — used for heartbeat and GetAssignment coroutines."""
    return aio.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)


# ── Gradient compression ──────────────────────────────────────────────────────

def compress_gradients(
    model:  nn.Module,
    topk_k: int,
) -> list[trainer_pb2.SparseGradient]:
    """
    Top-K sparsification per layer.

    For each parameter with a gradient:
      - If topk_k > 0: keep the topk_k elements with largest |value|.
      - If topk_k == 0: send all gradient elements (no compression).

    Gradients are moved to CPU + cast to float32 before sending so that
    MPS / CUDA tensors don't cause serialisation issues.
    """
    result: list[trainer_pb2.SparseGradient] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad  = param.grad.detach().cpu().float()
        shape = list(grad.shape)
        flat  = grad.flatten()
        n     = flat.numel()

        if topk_k > 0 and topk_k < n:
            _, top_idx = torch.topk(flat.abs(), topk_k)
            values     = flat[top_idx]
        else:
            top_idx = torch.arange(n, dtype=torch.int32)
            values  = flat

        result.append(trainer_pb2.SparseGradient(
            layer_name = name,
            indices    = top_idx.tolist(),
            values     = values.tolist(),
            shape      = shape,
        ))
    return result


# ── Weight application ────────────────────────────────────────────────────────

def load_full_weights(
    model:        nn.Module,
    weight_bytes: bytes,
    device:       torch.device,
) -> None:
    """Replace all model parameters with the leader's initial state dict."""
    buf   = io.BytesIO(weight_bytes)
    state = torch.load(buf, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    log.info(f"Loaded initial model weights ({len(weight_bytes) // 1024:,} KB)")


def apply_delta(
    model:  nn.Module,
    payload: bytes,
    device: torch.device,
) -> None:
    """Add the leader's weight delta (new - old) to every local parameter."""
    if not payload:
        return
    buf   = io.BytesIO(payload)
    delta = torch.load(buf, map_location="cpu", weights_only=True)
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, d in delta.items():
            if name in params:
                params[name].add_(d.to(device))


# ── Shared state (heartbeat ↔ training loop) ──────────────────────────────────

class WorkerSharedState:
    """
    Plain Python object acting as a thread-safe bulletin board.

    The heartbeat coroutine reads it; the training loop writes it.
    Python's GIL makes simple attribute reads/writes safe across threads
    for the primitive types used here (int, float, bool).
    """
    def __init__(self) -> None:
        self.status:  int   = trainer_pb2.IDLE
        self.loss:    float = -1.0
        self.steps:   int   = 0
        self.paused:  bool  = False
        self.stop:    bool  = False   # leader sent STOP or connection error


# ── Heartbeat coroutine (runs in asyncio) ─────────────────────────────────────

async def heartbeat_loop(
    stub:      trainer_pb2_grpc.TrainerServiceStub,   # async stub
    worker_id: str,
    shared:    WorkerSharedState,
) -> None:
    """
    Bidirectional heartbeat stream.
    Worker sends status every 5 s; leader replies with a command.
    Runs as an asyncio background task for the entire lifetime of the worker.
    """
    import asyncio as _asyncio

    async def _requests():
        while not shared.stop:
            yield trainer_pb2.HeartbeatRequest(
                worker_id       = worker_id,
                status          = shared.status,
                current_loss    = shared.loss,
                steps_completed = shared.steps,
                timestamp_utc   = int(time.time()),
            )
            await _asyncio.sleep(5)

    try:
        async for resp in stub.Heartbeat(_requests()):
            cmd = resp.command
            if cmd == trainer_pb2.HeartbeatResponse.STOP:
                log.warning("Leader sent STOP — halting training.")
                shared.stop = True
                return
            elif cmd == trainer_pb2.HeartbeatResponse.PAUSE:
                log.info("Leader sent PAUSE.")
                shared.paused = True
            elif cmd == trainer_pb2.HeartbeatResponse.CONTINUE:
                if shared.paused:
                    log.info("Leader sent CONTINUE — resuming.")
                shared.paused = False
    except _asyncio.CancelledError:
        pass
    except Exception as e:
        log.error(f"Heartbeat stream lost: {type(e).__name__}: {e}")
        shared.stop = True


# ── Training loop (runs in a thread, uses sync gRPC) ─────────────────────────

def training_loop(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    sync_stub:  trainer_pb2_grpc.TrainerServiceStub,   # sync stub
    worker_id:  str,
    config:     trainer_pb2.TrainingConfig,
    device:     torch.device,
    shared:     WorkerSharedState,
) -> None:
    """
    Synchronous training loop.  Runs in a thread pool so asyncio stays free.

    Each step:
      1. Forward pass + loss.backward()  (computes gradients on device)
      2. Top-K compress gradients        (move to CPU for serialisation)
      3. ExchangeGradients RPC           (blocking — waits for all workers)
      4. Apply weight delta from leader  (add delta to local params)
      5. Repeat
    """
    criterion = nn.CrossEntropyLoss()
    topk_k    = config.gradient_topk_k
    epochs    = config.total_epochs

    log.info(
        f"Training  epochs={epochs}  "
        f"batches/epoch≈{len(loader)}  "
        f"topk_k={topk_k}  device={device}"
    )

    for epoch in range(epochs):
        shared.status = trainer_pb2.TRAINING
        epoch_losses: list[float] = []

        for batch_idx, (images, labels) in enumerate(loader):
            if shared.stop:
                log.info("Stop flag set — exiting training loop.")
                return

            # Respect pause commands from leader
            while shared.paused and not shared.stop:
                time.sleep(0.5)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Forward + backward ────────────────────────────────────────
            model.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            loss_val          = loss.item()
            shared.loss       = loss_val
            shared.steps     += 1
            epoch_losses.append(loss_val)

            # ── Gradient compression ──────────────────────────────────────
            sparse_grads = compress_gradients(model, topk_k)

            # ── Push to leader, receive weight delta ──────────────────────
            try:
                update = sync_stub.ExchangeGradients(
                    trainer_pb2.GradientPush(
                        worker_id         = worker_id,
                        global_step       = shared.steps,
                        local_batch_count = images.size(0),
                        loss              = loss_val,
                        gradients         = sparse_grads,
                    )
                )
            except grpc.RpcError as exc:
                log.error(
                    f"ExchangeGradients failed: {exc.code()}  {exc.details()}"
                )
                shared.stop = True
                return

            # ── Apply weight delta from leader ────────────────────────────
            apply_delta(model, update.payload, device)

            if batch_idx % 20 == 0:
                log.info(
                    f"E{epoch + 1:02d}/{epochs}  "
                    f"B{batch_idx:04d}/{len(loader)}  "
                    f"loss={loss_val:.4f}  "
                    f"step={shared.steps}"
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        log.info(f"Epoch {epoch + 1}/{epochs} done  avg_loss={avg_loss:.4f}")

    shared.status = trainer_pb2.IDLE
    log.info("Training complete.")


# ── Proto builder helpers ─────────────────────────────────────────────────────

def _hw_to_proto(hw) -> trainer_pb2.HardwareInfo:
    _type_map = {
        AccelType.CUDA: trainer_pb2.AcceleratorInfo.CUDA,
        AccelType.MPS:  trainer_pb2.AcceleratorInfo.MPS,
        AccelType.CPU:  trainer_pb2.AcceleratorInfo.CPU,
    }
    accels = [
        trainer_pb2.AcceleratorInfo(
            type      = _type_map.get(a.type, trainer_pb2.AcceleratorInfo.CPU),
            name      = a.name,
            vram_gb   = a.vram_gb,
            gpu_cores = a.gpu_cores,
            tflops    = a.tflops,
        )
        for a in hw.accelerators
    ]
    return trainer_pb2.HardwareInfo(
        hostname       = hw.hostname,
        os             = hw.os,
        python_version = hw.python_version,
        torch_version  = hw.torch_version,
        cpu_cores      = hw.cpu_cores,
        ram_gb         = hw.ram_gb,
        accelerators   = accels,
    )


def _resolve_device(suggested: str) -> torch.device:
    """
    Trust the leader's device suggestion only if that device actually exists
    on this machine; fall back gracefully rather than crash.
    """
    if suggested.startswith("cuda"):
        if torch.cuda.is_available():
            idx = int(suggested.split(":")[-1]) if ":" in suggested else 0
            if idx < torch.cuda.device_count():
                return torch.device(f"cuda:{idx}")
        log.warning("Leader suggested CUDA but no CUDA device found — falling back.")
    if suggested == "mps":
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        log.warning("Leader suggested MPS but MPS unavailable — falling back.")
    return torch.device("cpu")


# ── Main entry point ──────────────────────────────────────────────────────────

async def run(cfg: argparse.Namespace) -> None:
    import asyncio

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
    if cfg.dry_run:
        log.info("Dry-run mode: skipping dataset download (synthetic data will be used)")
    else:
        log.info("Checking dataset …")
        await asyncio.get_event_loop().run_in_executor(None, ensure_dataset)

    # ── Step 3: open gRPC channels ────────────────────────────────────────────
    addr = f"{cfg.leader}:{cfg.port}"
    log.info(f"Connecting to leader at {addr} …")

    # Async channel: heartbeat (bidi stream) + GetAssignment
    async_ch  = _async_channel(cfg.leader, cfg.port)
    async_stub = trainer_pb2_grpc.TrainerServiceStub(async_ch)

    # Sync channel: ExchangeGradients (called from training thread)
    sync_ch   = _sync_channel(cfg.leader, cfg.port)
    sync_stub = trainer_pb2_grpc.TrainerServiceStub(sync_ch)

    # ── Step 4: register ──────────────────────────────────────────────────────
    reg = await async_stub.Register(
        trainer_pb2.RegisterRequest(
            hw_info   = _hw_to_proto(hw),
            benchmark = trainer_pb2.BenchmarkResult(
                score          = bm.score,
                forward_ms     = bm.forward_ms,
                memory_free_gb = bm.memory_free_gb,
            ),
        )
    )
    if not reg.accepted:
        log.error(f"Registration rejected: {reg.reject_reason}")
        await async_ch.close()
        return

    worker_id = reg.worker_id
    log.info(f"Registered as worker_id={worker_id}")

    # ── Step 5: shared state + background heartbeat ───────────────────────────
    shared    = WorkerSharedState()
    hb_task   = asyncio.create_task(
        heartbeat_loop(async_stub, worker_id, shared)
    )

    # ── Step 6: GetAssignment (blocks until leader types "start") ─────────────
    log.info("Waiting for assignment from leader …  (leader must type 'start')")
    shared.status = trainer_pb2.IDLE

    assignment_resp = await async_stub.GetAssignment(
        trainer_pb2.GetAssignmentRequest(worker_id=worker_id)
    )

    assignment = assignment_resp.assignment
    config     = assignment_resp.config
    device     = _resolve_device(assignment.primary_device)

    log.info(
        f"Assignment received:  "
        f"{len(assignment.indices):,} samples  "
        f"batch={assignment.local_batch_size}  "
        f"device={device}"
    )

    # ── Step 7: build model + load initial weights ────────────────────────────
    _MODEL_FNS = {
        "resnet18":  models.resnet18,
        "resnet50":  models.resnet50,
        "resnet101": models.resnet101,
    }
    model_fn = _MODEL_FNS.get(config.model_name, models.resnet101)
    log.info(f"Building {config.model_name} ({config.num_classes} classes) on {device} …")
    model = model_fn(weights=None, num_classes=config.num_classes)
    load_full_weights(model, assignment_resp.model_weights, device)
    model = model.to(device).train()

    # ── Step 8: build dataloader for this worker's shard ──────────────────────
    if cfg.dry_run:
        n_synthetic = min(len(assignment.indices), 320)   # enough for ~10 batches
        loader = _make_synthetic_loader(n_synthetic, assignment.local_batch_size, config.num_classes)
        log.info(f"Synthetic DataLoader: {n_synthetic} samples  batch={assignment.local_batch_size}")
    else:
        loader = make_train_loader(
            root       = None,
            indices    = list(assignment.indices),
            batch_size = assignment.local_batch_size,
            cpu_cores  = hw.cpu_cores,
        )
    log.info(
        f"DataLoader ready:  {len(loader.dataset):,} samples  "
        f"batch={assignment.local_batch_size}  "
        f"batches/epoch={len(loader)}"
    )

    # ── Step 9: training loop in thread pool ──────────────────────────────────
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: training_loop(
                model, loader, sync_stub,
                worker_id, config, device, shared,
            ),
        )
    finally:
        # Graceful cleanup
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
        await async_ch.close()
        sync_ch.close()
        log.info("Worker shut down.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    p = argparse.ArgumentParser(description="Distributed ResNet-101 — Worker")
    p.add_argument(
        "--leader", required=True,
        help="Leader hostname or Tailscale magic DNS  "
             "(e.g. leader-macbook-pro.taila5426e.ts.net)"
    )
    p.add_argument("--port",  type=int, default=50051)
    p.add_argument(
        "--cache-dir", default=None, dest="cache_dir",
        help="Override dataset cache directory (default: ~/.cache/tiny-imagenet-200)"
    )
    p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Skip dataset download and use synthetic tensors — useful for protocol testing"
    )
    cfg = p.parse_args()

    if cfg.cache_dir:
        _CACHE_ROOT = Path(cfg.cache_dir)

    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(0)
