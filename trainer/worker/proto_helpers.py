"""
trainer/worker/proto_helpers.py — Protobuf / device conversion utilities.
"""
from __future__ import annotations

import torch

from proto import trainer_pb2
from trainer.utils.hardware import AccelType
from trainer.core.logging import get as _get_log

log = _get_log("worker")


def hw_to_proto(hw) -> trainer_pb2.HardwareInfo:
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


def resolve_device(suggested: str) -> torch.device:
    """
    Trust the leader's device suggestion only if that device exists locally;
    falls back gracefully rather than crashing.
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


def parse_dataset_list(value: str | None, primary: str) -> list[str]:
    valid = {"tinyimagenet", "cifar10"}
    names: list[str] = []

    for raw in (value or primary).split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name == "all":
            for candidate in ("tinyimagenet", "cifar10"):
                if candidate not in names:
                    names.append(candidate)
            continue
        if name not in valid:
            raise ValueError(
                f"Unsupported dataset {name!r}; expected tinyimagenet, cifar10, or all"
            )
        if name not in names:
            names.append(name)

    if primary not in names:
        names.insert(0, primary)
    return names
