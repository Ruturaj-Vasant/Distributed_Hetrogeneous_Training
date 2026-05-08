# Distributed Heterogeneous Training — ResNet-101 on Tiny ImageNet-200

Train ResNet-101 across multiple laptops (Mac MPS + Windows CUDA) connected over **Tailscale**, with dynamic worker registration, hardware-proportional data sharding, and Top-K gradient compression.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TAILSCALE VPN                         │
│                                                         │
│  ┌─────────────────────────────────────┐                │
│  │  LEADER  (this Mac)                 │                │
│  │  leader-macbook-pro.taila…ts.net    │                │
│  │                                     │                │
│  │  • gRPC server :50051               │                │
│  │  • Parameter server (holds model)   │                │
│  │  • Shard allocator (score-based)    │                │
│  │  • Gradient aggregator              │                │
│  └──────────┬──────────────┬───────────┘                │
│             │ gRPC streams │                             │
│    ┌────────┴──────┐  ┌────┴──────────┐                 │
│    │ Worker (Mac)  │  │ Worker (Win)  │                 │
│    │ Apple M-series│  │ NVIDIA GPU    │                 │
│    │ score: 1580   │  │ score: 4200   │                 │
│    │ 27% of data   │  │ 73% of data   │                 │
│    └───────────────┘  └───────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

Each worker:
1. Probes its own hardware and runs a microbenchmark
2. Registers with the leader, sending its capability score
3. Blocks on `GetAssignment` until the operator starts training
4. Receives its data shard (proportional to score), config, and initial model weights
5. Trains locally → compresses gradients (Top-K) → pushes to leader
6. Leader aggregates all workers' gradients, steps the optimizer, returns weight delta

---

## Quick Start

### Leader (this MacBook Pro)

```bash
# 1. Install dependencies (once)
pip3 install -r requirements.txt
./generate_proto.sh

# 2. Start the leader server
python3 leader.py --min-workers 1

# 3. When all workers have registered, type:
leader> start

# Other commands:
leader> status   # live worker table
leader> quit
```

**Optional flags:**
```bash
python3 leader.py \
  --port 50051 \
  --min-workers 2 \
  --auto-start \        # start automatically when min-workers threshold met
  --model resnet101 \   # resnet18 | resnet50 | resnet101
  --epochs 90 \
  --lr 0.1 \
  --topk 50000          # Top-K gradient elements per layer (0 = no compression)
```

---

### Worker — Mac (one command)

```bash
LEADER_HOST=leader-macbook-pro.taila5426e.ts.net bash <(curl -fsSL \
  https://raw.githubusercontent.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training/main/scripts/worker_macos.sh)
```

Or if you have the repo already:
```bash
LEADER_HOST=leader-macbook-pro.taila5426e.ts.net bash scripts/worker_macos.sh
```

---

### Worker — Windows (one command, run PowerShell as Administrator)

```powershell
$env:LEADER_HOST="leader-macbook-pro.taila5426e.ts.net"
irm https://raw.githubusercontent.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training/main/scripts/worker_windows.ps1 | iex
```

Or if you have the repo already:
```powershell
$env:LEADER_HOST="leader-macbook-pro.taila5426e.ts.net"
.\scripts\worker_windows.ps1
```

The Windows script **auto-detects your CUDA version** and installs the matching PyTorch build (CUDA 11.8 / 12.1 / 12.4 or CPU-only fallback).

---

## What the worker scripts do

| Step | Mac script | Windows script |
|------|-----------|----------------|
| Prerequisites | Xcode CLT, Homebrew | winget |
| Python | 3.11 via Homebrew | 3.11 via winget |
| Git | Homebrew | winget |
| Tailscale | Homebrew + browser auth | winget + browser auth |
| Repo | `git clone` or `git pull` | `git clone` or `git pull` |
| Virtual env | `python3.11 -m venv` | `py -3.11 -m venv` |
| PyTorch | MPS-enabled (default pip) | CUDA or CPU (auto-detected) |
| Proto stubs | `generate_proto.sh` | inline protoc call |
| Dataset | `dataset.py` (~236 MB) | `dataset.py` (~236 MB) |
| Launch | `python3 worker.py` | `python worker.py` |

---

## Manual setup (without the bootstrap scripts)

```bash
# All machines:
git clone https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git
cd Distributed_Hetrogeneous_Training
pip3 install -r requirements.txt
./generate_proto.sh
python3 dataset.py          # downloads ~236 MB, skippable (worker auto-downloads)

# Workers:
python3 worker.py --leader leader-macbook-pro.taila5426e.ts.net
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEADER_HOST` | `leader-macbook-pro.taila5426e.ts.net` | Leader Tailscale DNS |
| `LEADER_PORT` | `50051` | gRPC port |
| `REPO_URL` | GitHub HTTPS URL | Override clone source |
| `REPO_DIR` | `~/distributed-resnet` | Local clone path |
| `SKIP_DATASET` | `0` | Set to `1` to skip pre-download |

---

## Project layout

```
distributed-resnet/
├── leader.py             # gRPC parameter server + interactive CLI
├── worker.py             # Worker: probe → register → train → push gradients
├── hardware_probe.py     # Cross-platform HW detection + capability score
├── dataset.py            # Tiny ImageNet-200 download, val-set setup, DataLoaders
├── test_e2e.py           # End-to-end smoke test (no dataset, no Tailscale needed)
├── generate_proto.sh     # Regenerate proto/trainer_pb2*.py after editing .proto
├── requirements.txt
├── proto/
│   ├── trainer.proto     # gRPC service + message definitions
│   ├── trainer_pb2.py    # generated
│   └── trainer_pb2_grpc.py # generated
└── scripts/
    ├── worker_macos.sh   # Mac one-shot bootstrap
    └── worker_windows.ps1 # Windows one-shot bootstrap
```

---

## How sharding works

Each worker sends a **capability score** computed from:
- CPU cores and RAM
- GPU VRAM (CUDA) or GPU core count (MPS)
- Actual forward-pass latency (microbenchmark on ResNet-18)

The leader distributes dataset indices proportionally:

```
worker_A_fraction = worker_A_score / sum(all_scores)
worker_A_samples  = int(worker_A_fraction × 100,000)
```

A Windows laptop with an RTX 3060 (score ≈ 4200) paired with a MacBook M3 Pro (score ≈ 1580) would receive **73% / 27%** of the data respectively.

---

## Running the smoke test

No Tailscale, no dataset, no extra machines needed:

```bash
python3 test_e2e.py
```

Verifies the full protocol — registration, assignment, gradient round-trips, weight delta application — in ~15 seconds.
