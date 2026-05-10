"""trainer/dashboard/server.py — FastAPI + WebSocket dashboard server."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from trainer.leader.service import LeaderService

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="dtrain", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

_service: "LeaderService | None" = None
_clients: Set[WebSocket] = set()
_loss_history: list[float] = []          # server-side buffer — survives page refresh
_MAX_HISTORY = 2000

_STATUS_NAMES = {0: "idle", 1: "training"}


def _snapshot(svc: "LeaderService") -> dict:
    now = time.monotonic()
    workers = []
    for w in svc._workers.values():
        ago = round(now - w.last_heartbeat, 1)
        workers.append({
            "worker_id":   w.worker_id,
            "hostname":    w.hostname,
            "os":          w.os_name,
            "status":      _STATUS_NAMES.get(w.status, "idle"),
            "score":       round(w.score, 1),
            "shard_size":  len(w.shard_indices),
            "last_loss":   round(w.last_loss, 4),
            "steps":       w.steps,
            "accel":       w.accel_summary,
            "heartbeat_ago": ago,
            "alive":       w.is_alive,
            "assigned":    w.assigned,
        })

    pending = []
    for wid in svc._pending_worker_ids:
        w = svc._workers.get(wid)
        if w:
            pending.append({
                "worker_id": w.worker_id,
                "hostname":  w.hostname,
                "score":     round(w.score, 1),
                "accel":     w.accel_summary,
            })

    cfg = svc.cfg
    return {
        "type":    "state",
        "ts":      time.time(),
        "phase":   "training" if svc._training_started.is_set() else "waiting",
        "step":    svc._global_step,
        "epoch":   svc._current_epoch,
        "loss":    round(svc._global_loss, 4),
        "workers": workers,
        "pending": pending,
        "cfg": {
            "model":      getattr(cfg, "model_name", "resnet101"),
            "dataset":    getattr(cfg, "dataset", "tinyimagenet"),
            "epochs":     getattr(cfg, "epochs", 90),
            "lr":         getattr(cfg, "lr", 0.1),
            "topk":       getattr(cfg, "topk", 50000),
            "batch_size": getattr(cfg, "batch_size", 32),
        },
    }


async def _broadcast_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
        if _service is None:
            continue
        snap = _snapshot(_service)
        # Accumulate loss on the server side so refresh doesn't lose history
        if snap["loss"] > 0:
            _loss_history.append(snap["loss"])
            if len(_loss_history) > _MAX_HISTORY:
                _loss_history.pop(0)
        if not _clients:
            continue
        msg = json.dumps(snap)
        dead: Set[WebSocket] = set()
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text()


@app.get("/leader", response_class=HTMLResponse)
async def leader_page():
    return (_STATIC / "leader.html").read_text()


@app.get("/worker", response_class=HTMLResponse)
async def worker_page():
    return (_STATIC / "worker.html").read_text()


@app.get("/watch", response_class=HTMLResponse)
async def watch_page():
    return (_STATIC / "watch.html").read_text()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        if _service is not None:
            await ws.send_text(json.dumps(_snapshot(_service)))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


@app.get("/api/history")
async def api_history():
    """Return buffered loss history so the chart survives page refresh."""
    return {"loss": _loss_history}


@app.post("/api/start")
async def api_start():
    if _service is None:
        return {"ok": False, "error": "not ready"}
    await _service.start_training()
    return {"ok": True}


@app.post("/api/reset")
async def api_reset():
    if _service is None:
        return {"ok": False, "error": "not ready"}
    await _service.reset_training()
    return {"ok": True}


@app.post("/api/admit")
async def api_admit(body: dict = {}):
    if _service is None:
        return {"ok": False, "error": "not ready"}
    from proto import trainer_pb2
    ids = body.get("worker_ids", [])
    resp = await _service.AdmitWorkers(
        trainer_pb2.AdmitWorkersRequest(worker_ids=ids), context=None
    )
    return {
        "ok":        True,
        "admitted":  list(resp.admitted_ids),
        "not_found": list(resp.not_found_ids),
    }


async def serve(service: "LeaderService", host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the dashboard alongside the gRPC server. Non-blocking — runs as asyncio tasks."""
    global _service
    _service = service

    import uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    uv_server = uvicorn.Server(config)

    asyncio.create_task(_broadcast_loop())
    asyncio.create_task(uv_server.serve())
