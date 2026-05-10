"""
trainer/worker/gradients.py — Gradient compression, weight apply, weight load.
Pure tensor operations — no network calls.
"""
from __future__ import annotations
import io

import torch
import torch.nn as nn

from proto import trainer_pb2
from trainer.core.logging import get as _get_log

log = _get_log("worker")


def compress_gradients(
    model:  nn.Module,
    topk_k: int,
) -> list[trainer_pb2.SparseGradient]:
    """
    Top-K sparsification per layer.

    - topk_k > 0: keep the topk_k elements with largest |value| per layer.
    - topk_k == 0: send all gradient elements (no compression).

    Flushes the device command queue once before the per-layer loop so that
    MPS/CUDA tensors don't trigger an implicit full-device sync per layer.
    """
    result: list[trainer_pb2.SparseGradient] = []

    params_with_grad = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    if params_with_grad:
        dev = params_with_grad[0][1].grad.device
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize(dev)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad  = param.grad.detach().float()
        shape = list(grad.shape)
        flat  = grad.flatten()
        n     = flat.numel()

        if topk_k > 0 and topk_k < n:
            _, top_idx = torch.topk(flat.abs(), topk_k)
            values     = flat[top_idx].cpu()
            top_idx    = top_idx.to(torch.int32).cpu()
        else:
            top_idx = torch.arange(n, dtype=torch.int32)
            values  = flat.cpu()

        result.append(trainer_pb2.SparseGradient(
            layer_name = name,
            indices    = top_idx.tolist(),
            values     = values.tolist(),
            shape      = shape,
        ))
    return result


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
    model:   nn.Module,
    payload: bytes,
    device:  torch.device,
) -> None:
    """Add the leader's weight delta (new − old, float16) to every local parameter."""
    if not payload:
        return
    buf   = io.BytesIO(payload)
    delta = torch.load(buf, map_location="cpu", weights_only=True)
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, d in delta.items():
            if name in params:
                params[name].add_(d.to(device))
