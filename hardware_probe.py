"""
hardware_probe.py

Detects hardware on the current machine and computes a capability score.
Returns plain dataclasses — no gRPC dependency — so this can be tested standalone.
"""

import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AccelType(str, Enum):
    CPU  = "cpu"
    CUDA = "cuda"
    MPS  = "mps"


@dataclass
class AcceleratorInfo:
    type:      AccelType
    name:      str
    vram_gb:   float = 0.0   # unified memory devices report 0 here
    gpu_cores: int   = 0
    tflops:    float = 0.0


@dataclass
class HardwareInfo:
    hostname:       str
    os:             str
    python_version: str
    torch_version:  str
    cpu_cores:      int
    ram_gb:         float
    accelerators:   list[AcceleratorInfo] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    score:          float   # composite (higher = more capable)
    forward_ms:     float   # latency of microbenchmark forward pass
    memory_free_gb: float   # free memory on primary accelerator


@dataclass
class ProbeResult:
    hw:        HardwareInfo
    benchmark: BenchmarkResult


# ─────────────────────────────────────────────────────────────────────────────
#  Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_hardware() -> HardwareInfo:
    import torch

    hostname = socket.gethostname()
    os_name  = platform.system().lower()   # "darwin", "windows", "linux"

    # CPU / RAM
    cpu_cores = _logical_cpu_count()
    ram_gb    = _total_ram_gb()

    accelerators: list[AcceleratorInfo] = []

    # CUDA GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024 ** 3)
            # Rough FP32 TFLOPS from SM count and clock when PyTorch exposes
            # clock data. Some Windows/CUDA builds omit max_clock_rate.
            max_clock_rate = getattr(props, "max_clock_rate", 0) or 0
            tflops = (props.multi_processor_count * max_clock_rate * 1e-6 * 2) / 1e3
            accelerators.append(AcceleratorInfo(
                type      = AccelType.CUDA,
                name      = props.name,
                vram_gb   = round(vram, 2),
                gpu_cores = props.multi_processor_count * 128,  # approx CUDA cores
                tflops    = round(tflops, 2),
            ))

    # Apple MPS
    elif _mps_available():
        info = _mps_info()
        accelerators.append(info)

    # Always include CPU as fallback accelerator entry
    accelerators.append(AcceleratorInfo(
        type      = AccelType.CPU,
        name      = platform.processor() or "Unknown CPU",
        gpu_cores = 0,
        vram_gb   = 0.0,
        tflops    = 0.0,
    ))

    return HardwareInfo(
        hostname       = hostname,
        os             = os_name,
        python_version = sys.version.split()[0],
        torch_version  = torch.__version__,
        cpu_cores      = cpu_cores,
        ram_gb         = round(ram_gb, 2),
        accelerators   = accelerators,
    )


def _logical_cpu_count() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=True)
    except ImportError:
        import os
        return os.cpu_count() or 1


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _mps_info() -> AcceleratorInfo:
    """
    Apple doesn't expose GPU core count via PyTorch; use subprocess to query
    system_profiler on macOS. Falls back to zero if unavailable.
    """
    gpu_cores = 0
    name      = "Apple GPU (MPS)"
    try:
        import subprocess, json
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            timeout=5,
        )
        data = json.loads(out)
        displays = data.get("SPDisplaysDataType", [])
        if displays:
            d = displays[0]
            name      = d.get("sppci_model", name)
            cores_str = d.get("sppci_cores", "0")
            gpu_cores = int(cores_str) if str(cores_str).isdigit() else 0
    except Exception:
        pass

    return AcceleratorInfo(
        type      = AccelType.MPS,
        name      = name,
        vram_gb   = 0.0,   # shares system RAM
        gpu_cores = gpu_cores,
        tflops    = 0.0,   # not reliably available
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Microbenchmark  (ResNet-18 forward on 16 images — fast but representative)
# ─────────────────────────────────────────────────────────────────────────────

_BENCH_BATCH = 16
_BENCH_WARMUP = 3
_BENCH_RUNS   = 10


def _run_benchmark(device: "torch.device") -> tuple[float, float]:
    """Returns (avg_forward_ms, free_memory_gb)."""
    import torch
    import torchvision.models as models

    model = models.resnet18(weights=None).to(device).eval()
    dummy = torch.randn(_BENCH_BATCH, 3, 64, 64, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(_BENCH_WARMUP):
            _ = model(dummy)

    _sync(device)

    # Timed runs
    times: list[float] = []
    with torch.no_grad():
        for _ in range(_BENCH_RUNS):
            t0 = time.perf_counter()
            _  = model(dummy)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    free_gb = _free_memory_gb(device)

    del model, dummy
    return avg_ms, free_gb


def _sync(device: "torch.device") -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _free_memory_gb(device: "torch.device") -> float:
    import torch
    try:
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return free / (1024 ** 3)
        elif device.type == "mps":
            # MPS uses unified memory; report free system RAM as proxy
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _primary_device() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────────

# Weights are intentionally tunable — change without touching the formula shape.
_W_CPU_CORE  = 0.5
_W_RAM       = 0.1
_W_CUDA_VRAM = 8.0
_W_MPS_CORES = 0.3
_W_LATENCY   = 50.0  # contribution from benchmark latency (lower ms → higher score)


def _compute_score(hw: HardwareInfo, forward_ms: float) -> float:
    score = hw.cpu_cores * _W_CPU_CORE + hw.ram_gb * _W_RAM

    for accel in hw.accelerators:
        if accel.type == AccelType.CUDA:
            score += accel.vram_gb * _W_CUDA_VRAM
        elif accel.type == AccelType.MPS:
            score += accel.gpu_cores * _W_MPS_CORES

    # Latency bonus: treat 100ms as baseline; faster = higher bonus
    if forward_ms > 0:
        score += _W_LATENCY * (100.0 / forward_ms)

    return round(score, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def probe() -> ProbeResult:
    """
    Run full hardware detection + benchmark.
    Safe to call on macOS (MPS), Windows (CUDA), Linux (CUDA/CPU).
    """
    hw     = _detect_hardware()
    device = _primary_device()

    print(f"[probe] Running benchmark on device: {device} ...")
    forward_ms, free_gb = _run_benchmark(device)

    score = _compute_score(hw, forward_ms)

    benchmark = BenchmarkResult(
        score          = score,
        forward_ms     = round(forward_ms, 2),
        memory_free_gb = round(free_gb, 2),
    )
    return ProbeResult(hw=hw, benchmark=benchmark)


def probe_to_dict(result: Optional[ProbeResult] = None) -> dict:
    """Serialise ProbeResult to a plain dict (for logging / gRPC construction)."""
    r = result or probe()
    return {
        "hw": {
            "hostname":       r.hw.hostname,
            "os":             r.hw.os,
            "python_version": r.hw.python_version,
            "torch_version":  r.hw.torch_version,
            "cpu_cores":      r.hw.cpu_cores,
            "ram_gb":         r.hw.ram_gb,
            "accelerators": [
                {
                    "type":      a.type.value,
                    "name":      a.name,
                    "vram_gb":   a.vram_gb,
                    "gpu_cores": a.gpu_cores,
                    "tflops":    a.tflops,
                }
                for a in r.hw.accelerators
            ],
        },
        "benchmark": {
            "score":          r.benchmark.score,
            "forward_ms":     r.benchmark.forward_ms,
            "memory_free_gb": r.benchmark.memory_free_gb,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Hardware Probe")
    print("=" * 60)

    result = probe()
    data   = probe_to_dict(result)

    print(json.dumps(data, indent=2))
    print()
    print(f"  Composite score : {result.benchmark.score}")
    print(f"  Benchmark       : {result.benchmark.forward_ms} ms / forward pass")
    print(f"  Free memory     : {result.benchmark.memory_free_gb} GB")
    print(f"  Primary device  : {_primary_device()}")
