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

# ── Self-bootstrap: create .venv, install deps, generate proto stubs ──────────
# Uses only stdlib so this block runs before any third-party imports.

def _bootstrap() -> None:
    import os, subprocess, sys
    from pathlib import Path

    proj = Path(__file__).resolve().parent
    venv = proj / ".venv"
    is_win = sys.platform == "win32"
    venv_py = venv / ("Scripts/python.exe" if is_win else "bin/python")

    # Already inside a venv — nothing to do.
    if sys.prefix != sys.base_prefix:
        return

    # Create venv if it doesn't exist yet.
    if not venv_py.exists():
        print("[bootstrap] Creating .venv …", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    # Install / verify dependencies.
    probe = subprocess.run(
        [str(venv_py), "-c", "import torch, torchvision, grpc"],
        capture_output=True,
    )
    if probe.returncode != 0:
        print("[bootstrap] Installing dependencies …", flush=True)
        subprocess.run(
            [str(venv_py), "-m", "pip", "install",
             "-r", str(proj / "requirements.txt"), "-q"],
            check=True,
        )

    # Generate gRPC proto stubs if missing.
    proto_dir = proj / "proto"
    if not (proto_dir / "trainer_pb2.py").exists():
        print("[bootstrap] Generating gRPC proto stubs …", flush=True)
        subprocess.run(
            [
                str(venv_py), "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={proto_dir}",
                f"--grpc_python_out={proto_dir}",
                str(proto_dir / "trainer.proto"),
            ],
            check=True,
        )
        grpc_f = proto_dir / "trainer_pb2_grpc.py"
        if grpc_f.exists():
            grpc_f.write_text(
                grpc_f.read_text().replace(
                    "import trainer_pb2", "from . import trainer_pb2", 1
                )
            )
        init = proto_dir / "__init__.py"
        if not init.exists():
            init.write_text("")

    # Re-exec this script inside the venv.
    print("[bootstrap] Launching inside .venv …", flush=True)
    if is_win:
        result = subprocess.run([str(venv_py)] + sys.argv)
        sys.exit(result.returncode)
    else:
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)


_bootstrap()

# ── Imports (guaranteed to succeed — either already in venv, or re-exec'd) ───

import argparse
import asyncio
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


def _rpc_error_summary(exc: BaseException) -> str:
    if isinstance(exc, grpc.aio.AioRpcError):
        return f"{exc.code().name}: {exc.details()}"
    if isinstance(exc, asyncio.TimeoutError):
        return "timed out waiting for gRPC channel readiness"
    return f"{type(exc).__name__}: {exc}"


async def _connect_and_register(
    cfg: argparse.Namespace,
    hw,
    bm,
) -> tuple[
    aio.Channel,
    trainer_pb2_grpc.TrainerServiceStub,
    grpc.Channel,
    trainer_pb2_grpc.TrainerServiceStub,
    trainer_pb2.RegisterResponse,
]:
    """
    Build fresh channels and register, retrying transient Tailscale/gRPC startup
    failures instead of crashing the worker on the first unavailable response.
    """
    addr = f"{cfg.leader}:{cfg.port}"
    last_error = "unknown error"

    for attempt in range(1, cfg.connect_retries + 1):
        async_ch = _async_channel(cfg.leader, cfg.port)
        async_stub = trainer_pb2_grpc.TrainerServiceStub(async_ch)
        sync_ch = _sync_channel(cfg.leader, cfg.port)
        sync_stub = trainer_pb2_grpc.TrainerServiceStub(sync_ch)

        try:
            log.info(
                f"Connecting to leader at {addr} "
                f"(attempt {attempt}/{cfg.connect_retries}) …"
            )
            await asyncio.wait_for(
                async_ch.channel_ready(),
                timeout=cfg.connect_timeout,
            )
            reg = await async_stub.Register(
                trainer_pb2.RegisterRequest(
                    hw_info=_hw_to_proto(hw),
                    benchmark=trainer_pb2.BenchmarkResult(
                        score=bm.score,
                        forward_ms=bm.forward_ms,
                        memory_free_gb=bm.memory_free_gb,
                    ),
                ),
                timeout=cfg.rpc_timeout,
                wait_for_ready=True,
            )
            return async_ch, async_stub, sync_ch, sync_stub, reg
        except (asyncio.TimeoutError, grpc.aio.AioRpcError) as exc:
            last_error = _rpc_error_summary(exc)
            await async_ch.close()
            sync_ch.close()

            if attempt >= cfg.connect_retries:
                break

            delay = min(cfg.retry_backoff * attempt, cfg.max_retry_backoff)
            log.warning(
                f"Leader connection failed: {last_error}; "
                f"retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Could not register with leader at {addr} after "
        f"{cfg.connect_retries} attempt(s): {last_error}"
    )


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
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: ensure_dataset(cfg.cache_dir)
        )

    # ── Step 3/4: open gRPC channels and register ─────────────────────────────
    async_ch, async_stub, sync_ch, sync_stub, reg = await _connect_and_register(
        cfg, hw, bm
    )
    if not reg.accepted:
        log.error(f"Registration rejected: {reg.reject_reason}")
        await async_ch.close()
        sync_ch.close()
        return

    worker_id = reg.worker_id
    log.info(f"Registered as worker_id={worker_id}")

    # ── Step 5: shared state + background heartbeat ───────────────────────────
    shared    = WorkerSharedState()
    hb_task   = asyncio.create_task(
        heartbeat_loop(async_stub, worker_id, shared)
    )

    # ── Steps 6-9: training loop — repeats for each new job ──────────────────
    _MODEL_FNS = {
        "resnet18":  models.resnet18,
        "resnet50":  models.resnet50,
        "resnet101": models.resnet101,
    }

    try:
        while True:
            # Step 6: wait for assignment
            shared.status = trainer_pb2.IDLE
            shared.stop   = False
            shared.loss   = -1.0
            shared.steps  = 0
            log.info("Waiting for assignment from leader …  (leader must type 'start')")

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

            # Step 7: build model + load initial weights
            model_fn = _MODEL_FNS.get(config.model_name, models.resnet101)
            log.info(f"Building {config.model_name} ({config.num_classes} classes) on {device} …")
            model = model_fn(weights=None, num_classes=config.num_classes)
            load_full_weights(model, assignment_resp.model_weights, device)
            model = model.to(device).train()

            # Step 8: build dataloader
            if cfg.dry_run:
                n_synthetic = min(len(assignment.indices), 320)
                loader = _make_synthetic_loader(n_synthetic, assignment.local_batch_size, config.num_classes)
                log.info(f"Synthetic DataLoader: {n_synthetic} samples  batch={assignment.local_batch_size}")
            else:
                loader = make_train_loader(
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

            # Step 9: training loop in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: training_loop(
                    model, loader, sync_stub,
                    worker_id, config, device, shared,
                ),
            )

            if shared.stop:
                # Error or STOP command from leader — exit worker
                log.info("Stopping worker on leader instruction or training error.")
                break

            # Training completed normally — stay connected for next run
            log.info("Training complete. Staying connected — waiting for next job (leader: 'reset' then 'start').")

    finally:
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
        "--leader",
        default="leader-macbook-pro.taila5426e.ts.net",
        help="Leader hostname or Tailscale magic DNS "
             "(default: leader-macbook-pro.taila5426e.ts.net)"
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
    p.add_argument(
        "--connect-timeout", type=float, default=20.0, dest="connect_timeout",
        help="Seconds to wait for the gRPC channel to become ready per attempt"
    )
    p.add_argument(
        "--rpc-timeout", type=float, default=30.0, dest="rpc_timeout",
        help="Seconds to wait for registration RPC completion"
    )
    p.add_argument(
        "--connect-retries", type=int, default=6, dest="connect_retries",
        help="Number of leader connection/registration attempts before failing"
    )
    p.add_argument(
        "--retry-backoff", type=float, default=2.0, dest="retry_backoff",
        help="Base seconds between retries; multiplied by attempt number"
    )
    p.add_argument(
        "--max-retry-backoff", type=float, default=15.0, dest="max_retry_backoff",
        help="Maximum seconds to sleep between connection retries"
    )
    cfg = p.parse_args()

    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(0)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)
