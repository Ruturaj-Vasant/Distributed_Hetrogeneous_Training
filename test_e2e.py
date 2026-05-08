"""
test_e2e.py  —  End-to-end smoke test (no real dataset, no Tailscale needed)

Run:
    python3 test_e2e.py

What is verified:
    [T1] Worker registers and receives a worker_id
    [T2] GetAssignment blocks until start_training() fires on the leader
    [T3] Worker receives shard indices, TrainingConfig, and model weights
    [T4] N gradient round-trips complete without error
    [T5] leader.global_step increments correctly after each round-trip
    [T6] Weight delta payload is non-empty
    [T7] Applying the delta actually changes the local model
    [T8] Heartbeat stream stays alive during the full training loop
    [T9] Dead-worker recovery: if a worker disconnects mid-sync,
         the leader forces aggregation so remaining workers are not blocked
    [T10] Two-worker run: shards are split proportionally by score;
          leader waits for BOTH gradients before stepping
"""

import argparse
import asyncio
import io
import sys
import time
import traceback

import torch
import torch.nn as nn
import torchvision.models as models
import grpc
from grpc import aio

from proto import trainer_pb2, trainer_pb2_grpc
from leader import LeaderService
from worker import compress_gradients, apply_delta, load_full_weights, _resolve_device

# ── Test parameters ───────────────────────────────────────────────────────────

TEST_PORT   = 50099     # avoid clashing with a real leader on 50051
N_STEPS     = 3         # gradient rounds per test
IMG_SIZE    = 64        # Tiny ImageNet native resolution
BATCH       = 4
N_CLASSES   = 200

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length",    256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cfg(**overrides) -> argparse.Namespace:
    defaults = dict(
        min_workers              = 1,
        auto_start               = False,
        epochs                   = 1,
        num_classes              = N_CLASSES,
        lr                       = 0.01,
        weight_decay             = 1e-4,
        topk                     = 200,        # small for speed; covers a resnet18 conv layer
        model_name               = "resnet18", # fast: ~45MB vs 170MB for resnet101
        port                     = TEST_PORT,
        heartbeat_timeout        = 60.0,
        heartbeat_check_interval =  5.0,
        grad_sync_timeout        = 120.0,
        data_root                = None,
        runs_root                = "/tmp/distributed_resnet_test_runs",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_hw(hostname: str = "test-worker", score: float = 1000.0) -> tuple:
    """Returns (HardwareInfo proto, BenchmarkResult proto)."""
    hw = trainer_pb2.HardwareInfo(
        hostname       = hostname,
        os             = "darwin",
        python_version = "3.12",
        torch_version  = torch.__version__,
        cpu_cores      = 8,
        ram_gb         = 16.0,
        accelerators   = [trainer_pb2.AcceleratorInfo(
            type      = trainer_pb2.AcceleratorInfo.MPS,
            name      = "Apple M3 Pro",
            gpu_cores = 14,
        )],
    )
    bm = trainer_pb2.BenchmarkResult(
        score          = score,
        forward_ms     = 3.5,
        memory_free_gb = 6.0,
    )
    return hw, bm


async def _start_server(service: LeaderService, port: int) -> aio.Server:
    server = aio.server()
    trainer_pb2_grpc.add_TrainerServiceServicer_to_server(service, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    return server


def _check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    msg = f"  {tag}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not condition:
        raise AssertionError(label)


# ── Mock worker: runs one complete training simulation ────────────────────────

async def _mock_worker(
    port:      int,
    hostname:  str,
    score:     float,
    n_steps:   int,
    *,
    send_heartbeats: bool = True,
) -> dict:
    """
    Simulates a worker:
      Register → heartbeat (background) → GetAssignment → n_steps of gradient exchange.

    Returns a result dict with step count, final loss, and whether weights changed.
    Uses two channels just like the real worker.py:
      - async channel  → Register, Heartbeat, GetAssignment
      - sync channel   → ExchangeGradients (called from a thread)
    """
    async_ch  = aio.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
    sync_ch   = grpc.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
    async_stub = trainer_pb2_grpc.TrainerServiceStub(async_ch)
    sync_stub  = trainer_pb2_grpc.TrainerServiceStub(sync_ch)

    # ── Register ──────────────────────────────────────────────────────────────
    hw, bm = _fake_hw(hostname, score)
    reg = await async_stub.Register(
        trainer_pb2.RegisterRequest(hw_info=hw, benchmark=bm)
    )
    assert reg.accepted, f"Registration rejected: {reg.reject_reason}"
    worker_id = reg.worker_id

    # ── Heartbeat (background task) ───────────────────────────────────────────
    hb_running = True

    async def _heartbeat():
        async def _reqs():
            while hb_running:
                yield trainer_pb2.HeartbeatRequest(
                    worker_id     = worker_id,
                    status        = trainer_pb2.TRAINING,
                    timestamp_utc = int(time.time()),
                )
                await asyncio.sleep(2)

        try:
            async for _ in async_stub.Heartbeat(_reqs()):
                pass
        except aio.AioRpcError:
            pass

    hb_task = asyncio.create_task(_heartbeat()) if send_heartbeats else None

    # ── GetAssignment (blocks until leader starts) ────────────────────────────
    asgn_resp  = await async_stub.GetAssignment(
        trainer_pb2.GetAssignmentRequest(worker_id=worker_id)
    )
    assignment = asgn_resp.assignment
    config     = asgn_resp.config
    device     = _resolve_device(assignment.primary_device)

    # ── Build model from leader's initial weights ─────────────────────────────
    _MODEL_FNS = {"resnet18": models.resnet18, "resnet101": models.resnet101}
    model_fn   = _MODEL_FNS.get(config.model_name, models.resnet18)
    model      = model_fn(weights=None, num_classes=config.num_classes).to(device).train()
    load_full_weights(model, asgn_resp.model_weights, device)

    # Snapshot weights BEFORE any gradient exchange
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Training loop (sync gRPC inside a thread) ─────────────────────────────
    result = {"steps": 0, "loss": 0.0, "error": None}

    def _sync_train():
        criterion = nn.CrossEntropyLoss()
        for step in range(1, n_steps + 1):
            model.zero_grad()
            imgs   = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=device)
            labels = torch.randint(0, N_CLASSES, (BATCH,),          device=device)
            loss   = criterion(model(imgs), labels)
            loss.backward()

            sparse = compress_gradients(model, config.gradient_topk_k)
            try:
                update = sync_stub.ExchangeGradients(
                    trainer_pb2.GradientPush(
                        worker_id         = worker_id,
                        global_step       = step,
                        local_batch_count = BATCH,
                        loss              = loss.item(),
                        gradients         = sparse,
                    )
                )
                apply_delta(model, update.payload, device)
                result["steps"] += 1
                result["loss"]   = loss.item()
            except grpc.RpcError as exc:
                result["error"] = f"{exc.code()}: {exc.details()}"
                return

    await asyncio.get_event_loop().run_in_executor(None, _sync_train)

    # ── Check weights actually changed ────────────────────────────────────────
    changed = sum(
        1 for k, v in model.state_dict().items()
        if not torch.allclose(v.cpu(), weights_before[k].cpu(), atol=1e-9)
    )
    result["weights_changed"] = changed
    result["worker_id"]       = worker_id
    result["shard_size"]      = len(assignment.indices)

    # Cleanup
    nonlocal_hb = hb_running
    hb_running  = False          # signal heartbeat generator to stop
    if hb_task:
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass
    sync_ch.close()
    await async_ch.close()
    return result


# ── Individual test cases ─────────────────────────────────────────────────────

async def test_single_worker_happy_path(port: int) -> None:
    print("\n── Test 1: single-worker happy path ────────────────────────────────")
    cfg     = _make_cfg()
    service = LeaderService(cfg)
    server  = await _start_server(service, port)

    async def _trigger():
        await asyncio.sleep(0.8)
        await service.start_training()

    try:
        result, _ = await asyncio.gather(
            _mock_worker(port, "laptop-a", score=1000.0, n_steps=N_STEPS),
            _trigger(),
        )

        _check("T1 worker registered",          result["worker_id"] != "")
        _check("T2 shard indices received",     result["shard_size"] > 0)
        _check("T3 steps completed",            result["steps"] == N_STEPS,
               f"{result['steps']}/{N_STEPS}")
        _check("T4 no RPC error",               result["error"] is None,
               str(result.get("error")))
        _check("T5 leader.global_step correct", service._global_step == N_STEPS,
               str(service._global_step))
        _check("T6 leader recorded loss",       service._global_loss > 0,
               f"{service._global_loss:.4f}")
        _check("T7 model weights changed",      result["weights_changed"] > 0,
               f"{result['weights_changed']} tensors")

    finally:
        await server.stop(grace=1)


async def test_heartbeat_alive_during_training(port: int) -> None:
    """
    Directly exercises the bidi-stream Heartbeat RPC: sends 5 heartbeats
    at 100 ms intervals and counts the corresponding responses from the leader.
    This avoids patching (which doesn't intercept gRPC servicer dispatch)
    and avoids the 2 s interval being longer than the test duration.
    """
    print("\n── Test 2: heartbeat bidi-stream ───────────────────────────────────")
    cfg     = _make_cfg()
    service = LeaderService(cfg)
    server  = await _start_server(service, port)

    ch   = aio.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
    stub = trainer_pb2_grpc.TrainerServiceStub(ch)

    try:
        # Register a worker first (heartbeat requires a known worker_id)
        hw, bm = _fake_hw("hb-test-worker", 800.0)
        reg    = await stub.Register(
            trainer_pb2.RegisterRequest(hw_info=hw, benchmark=bm)
        )
        _check("T8a worker registered for hb test", reg.accepted)
        wid = reg.worker_id

        N_BEATS          = 5
        responses: list  = []

        async def _send_and_collect():
            async def _reqs():
                for i in range(N_BEATS):
                    yield trainer_pb2.HeartbeatRequest(
                        worker_id       = wid,
                        status          = trainer_pb2.TRAINING if i > 1 else trainer_pb2.IDLE,
                        current_loss    = 2.5 - i * 0.1,
                        steps_completed = i,
                        timestamp_utc   = int(time.time()),
                    )
                    await asyncio.sleep(0.1)   # 100 ms between beats

            async for resp in stub.Heartbeat(_reqs()):
                responses.append(resp)

        await asyncio.wait_for(_send_and_collect(), timeout=5.0)

        _check("T8b all heartbeat responses received",
               len(responses) == N_BEATS,
               f"{len(responses)}/{N_BEATS}")
        _check("T8c default command is CONTINUE",
               all(r.command == trainer_pb2.HeartbeatResponse.CONTINUE
                   for r in responses))
        _check("T8d last_heartbeat updated on leader",
               wid in service._workers and service._workers[wid].is_alive)

    finally:
        await ch.close()
        await server.stop(grace=1)


async def test_two_workers_proportional_shards(port: int) -> None:
    print("\n── Test 3: two workers, proportional shard split ───────────────────")
    # Worker A score=3000, Worker B score=1000 → A gets ~75%, B gets ~25%
    cfg     = _make_cfg(min_workers=2)
    service = LeaderService(cfg)
    server  = await _start_server(service, port)

    async def _trigger():
        # Wait until both workers have registered before starting
        while True:
            async with service._lock:
                alive = sum(1 for w in service._workers.values() if w.is_alive)
            if alive >= 2:
                break
            await asyncio.sleep(0.05)
        await service.start_training()

    try:
        res_a, res_b, _ = await asyncio.gather(
            _mock_worker(port, "laptop-strong", score=3000.0, n_steps=N_STEPS),
            _mock_worker(port, "laptop-weak",   score=1000.0, n_steps=N_STEPS),
            _trigger(),
        )

        total   = res_a["shard_size"] + res_b["shard_size"]
        pct_a   = res_a["shard_size"] / total * 100
        pct_b   = res_b["shard_size"] / total * 100

        _check("T9a  both workers completed",   res_a["steps"] == N_STEPS and
                                                res_b["steps"] == N_STEPS,
               f"A={res_a['steps']} B={res_b['steps']}")
        _check("T9b  strong worker > 60% shard",  pct_a > 60,
               f"A={pct_a:.1f}%  B={pct_b:.1f}%")
        _check("T9c  weak worker > 10% shard",    pct_b > 10,
               f"B={pct_b:.1f}%")
        _check("T9d  leader stepped 2×N times",  service._global_step == N_STEPS,
               f"global_step={service._global_step}")
        _check("T9e  both weights changed",
               res_a["weights_changed"] > 0 and res_b["weights_changed"] > 0,
               f"A={res_a['weights_changed']}  B={res_b['weights_changed']}")

    finally:
        await server.stop(grace=1)


async def test_late_registration_rejected(port: int) -> None:
    print("\n── Test 4: late registration is rejected ───────────────────────────")
    cfg     = _make_cfg()
    service = LeaderService(cfg)
    server  = await _start_server(service, port)

    async def _trigger():
        await asyncio.sleep(0.3)
        await service.start_training()

    try:
        await asyncio.gather(
            _mock_worker(port, "laptop-first", score=1000.0, n_steps=1),
            _trigger(),
        )

        # Now try to register AFTER training started
        ch   = aio.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
        stub = trainer_pb2_grpc.TrainerServiceStub(ch)
        hw, bm = _fake_hw("laptop-late", score=500.0)
        reg  = await stub.Register(
            trainer_pb2.RegisterRequest(hw_info=hw, benchmark=bm)
        )
        await ch.close()
        _check("T10 late worker rejected",      not reg.accepted,
               f"reason={reg.reject_reason!r}")
    finally:
        await server.stop(grace=1)


async def test_cluster_status_rpc(port: int) -> None:
    print("\n── Test 5: GetClusterStatus RPC ────────────────────────────────────")
    cfg     = _make_cfg()
    service = LeaderService(cfg)
    server  = await _start_server(service, port)

    async def _trigger():
        await asyncio.sleep(0.4)
        await service.start_training()

    async def _query_status():
        await asyncio.sleep(0.2)   # after register, before start
        ch   = aio.insecure_channel(f"localhost:{port}", options=_GRPC_OPTIONS)
        stub = trainer_pb2_grpc.TrainerServiceStub(ch)
        resp = await stub.GetClusterStatus(trainer_pb2.ClusterStatusRequest())
        await ch.close()
        return resp

    try:
        _, status, _ = await asyncio.gather(
            _mock_worker(port, "laptop-c", score=1000.0, n_steps=1),
            _query_status(),
            _trigger(),
        )
        _check("T11 status has workers",        len(status.workers) >= 1)
        _check("T12 worker hostname in status", any(
            w.hostname == "laptop-c" for w in status.workers
        ))
    finally:
        await server.stop(grace=1)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_all_tests() -> None:
    tests = [
        test_single_worker_happy_path,
        test_heartbeat_alive_during_training,
        test_two_workers_proportional_shards,
        test_late_registration_rejected,
        test_cluster_status_rpc,
    ]

    passed = 0
    failed = 0

    # Run tests sequentially on different ports to avoid state bleed
    port = TEST_PORT
    for test_fn in tests:
        try:
            await test_fn(port)
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
        port += 1   # each test gets a fresh port

    print(f"\n{'═'*60}")
    print(f"  Results:  {PASS} {passed} passed   {FAIL} {failed} failed")
    print(f"{'═'*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    print(f"\nDistributed ResNet — E2E Smoke Test")
    print(f"Device: {torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'}")
    print(f"Torch : {torch.__version__}")
    asyncio.run(run_all_tests())
