#!/usr/bin/env bash
# worker_windows.sh  —  One-shot bootstrap + launch for a Windows worker node
#                       Run inside Git Bash (comes with Git for Windows)
#
# Usage:
#   LEADER_HOST="leader-macbook-pro.taila5426e.ts.net" bash scripts/worker_windows.sh
#
# What this script does:
#   1.  Checks winget is available
#   2.  Installs Python 3.11 (if missing)
#   3.  Installs / authenticates Tailscale (if missing)
#   4.  Clones the repo (or pulls latest if already cloned)
#   5.  Creates a Python virtual environment
#   6.  Installs all Python dependencies
#       - Detects NVIDIA GPU and installs CUDA-enabled PyTorch automatically
#       - Falls back to CPU-only PyTorch if no GPU is found
#   7.  Generates gRPC proto stubs
#   8.  Downloads Tiny ImageNet-200 dataset (~236 MB, skipped if cached)
#   9.  Runs the worker
#
# Environment variables (all optional):
#   LEADER_HOST  — leader Tailscale DNS or IP  (default: leader-macbook-pro.taila5426e.ts.net)
#   LEADER_PORT  — gRPC port                   (default: 50051)
#   REPO_URL     — git clone URL
#   REPO_DIR     — local clone path            (default: ~/distributed-resnet)
#   SKIP_DATASET — set to 1 to skip dataset download

set -euo pipefail

LEADER_HOST="${LEADER_HOST:-leader-macbook-pro.taila5426e.ts.net}"
LEADER_PORT="${LEADER_PORT:-50051}"
REPO_URL="${REPO_URL:-https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git}"
REPO_DIR="${REPO_DIR:-${HOME}/distributed-resnet}"
VENV_DIR="${REPO_DIR}/.venv"
SKIP_DATASET="${SKIP_DATASET:-0}"

# ── Logging ───────────────────────────────────────────────────────────────────

log() { printf '\033[1;36m[worker:win]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[worker:win] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── Step 1: winget ────────────────────────────────────────────────────────────

ensure_winget() {
    have winget && { log "winget already present"; return; }
    err "winget not found. Install 'App Installer' from the Microsoft Store, then re-run."
}

# winget_install: treats "already at latest version" as success
winget_install() {
    local id="$1"
    log "Installing ${id} via winget …"
    local out
    out=$(winget install --id "$id" --exact \
        --accept-source-agreements --accept-package-agreements --silent 2>&1) || {
        if printf '%s\n' "$out" | grep -qiE "No newer package|already installed|No available upgrade"; then
            log "${id} is already at the latest version"
            return 0
        fi
        printf '%s\n' "$out" >&2
        err "winget install failed for: ${id}"
    }
    printf '%s\n' "$out"
}

# ── Step 2: Python 3.11 ───────────────────────────────────────────────────────

# Run Python 3.11 with the given arguments.
# Tries the Windows Python Launcher (py -3.11) first, then common direct paths.
python311() {
    if have py; then
        local ver
        ver=$(py -3.11 -c \
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" \
            2>/dev/null | tr -d '[:space:]' || true)
        if [ "${ver:-}" = "3.11" ]; then
            py -3.11 "$@"
            return 0
        fi
    fi
    local p
    for p in \
        "${HOME}/AppData/Local/Programs/Python/Python311/python.exe" \
        "/c/Program Files/Python311/python.exe" \
        "/c/Program Files/Python/Python311/python.exe" \
        "/c/ProgramData/Python/Python311/python.exe"; do
        if [ -x "$p" ]; then
            local ver
            ver=$("$p" -c \
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" \
                2>/dev/null | tr -d '[:space:]' || true)
            if [ "${ver:-}" = "3.11" ]; then
                "$p" "$@"
                return 0
            fi
        fi
    done
    return 1
}

ensure_python311() {
    if python311 --version >/dev/null 2>&1; then
        log "Python 3.11 already present ($(python311 --version 2>&1))"
        return
    fi
    winget_install "Python.Python.3.11"
    sleep 3
    python311 --version >/dev/null 2>&1 \
        || err "Python 3.11 not found after install — open a new Git Bash window and re-run."
    log "Python 3.11 ready: $(python311 --version 2>&1)"
}

# ── Step 3: Tailscale ─────────────────────────────────────────────────────────

tailscale_running() {
    have tailscale || return 1
    tailscale status --json 2>/dev/null \
        | grep -q '"BackendState"[[:space:]]*:[[:space:]]*"Running"'
}

ensure_tailscale() {
    if ! have tailscale; then
        winget_install "Tailscale.Tailscale"
        # Add the common install path in case PATH hasn't refreshed
        export PATH="${PATH}:/c/Program Files/Tailscale"
    else
        log "Tailscale already present"
    fi

    if tailscale_running; then log "Tailscale is authenticated"; return; fi

    log "Authenticate Tailscale — a browser window will open (or copy the URL below):"
    local out
    out=$(tailscale up --timeout=1s 2>&1 || true)
    printf '%s\n' "$out"
    local url
    url=$(printf '%s\n' "$out" | grep -Eo 'https://[^[:space:]]+' | head -1 || true)
    [ -n "$url" ] && start "$url" >/dev/null 2>&1 || true

    log "Waiting for Tailscale authentication …"
    until tailscale_running; do sleep 5; done
    log "Tailscale is authenticated"
}

# ── Step 4: Clone / update repo ───────────────────────────────────────────────

ensure_repo() {
    if [ -d "${REPO_DIR}/.git" ]; then
        log "Repo already cloned at ${REPO_DIR} — pulling latest"
        git -C "${REPO_DIR}" pull --ff-only
    else
        log "Cloning ${REPO_URL} → ${REPO_DIR}"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
}

# ── Step 5 & 6: Virtual environment + dependencies ───────────────────────────

# On Windows, venv puts the Python binary under Scripts/ not bin/
venv_py() { printf '%s' "${VENV_DIR}/Scripts/python"; }

cuda_index_url() {
    local smi
    smi=$(command -v nvidia-smi.exe 2>/dev/null || command -v nvidia-smi 2>/dev/null || true)
    if [ -z "$smi" ]; then
        log "No NVIDIA GPU detected — installing CPU-only PyTorch"
        echo "https://download.pytorch.org/whl/cpu"; return
    fi
    local out major minor
    out=$("$smi" 2>/dev/null | grep "CUDA Version" || true)
    major=$(printf '%s' "$out" | grep -Eo '[0-9]+\.[0-9]+' | cut -d. -f1 | head -1 || echo 0)
    minor=$(printf '%s' "$out" | grep -Eo '[0-9]+\.[0-9]+' | cut -d. -f2 | head -1 || echo 0)
    log "CUDA ${major}.${minor} detected"
    if   [ "$major" -ge 12 ] && [ "$minor" -ge 4 ]; then echo "https://download.pytorch.org/whl/cu124"
    elif [ "$major" -ge 12 ] && [ "$minor" -ge 1 ]; then echo "https://download.pytorch.org/whl/cu121"
    elif [ "$major" -ge 11 ] && [ "$minor" -ge 8 ]; then echo "https://download.pytorch.org/whl/cu118"
    else
        log "CUDA ${major}.${minor} < 11.8 — falling back to CPU-only PyTorch"
        echo "https://download.pytorch.org/whl/cpu"
    fi
}

ensure_venv() {
    if [ ! -x "$(venv_py)" ]; then
        log "Creating virtual environment at ${VENV_DIR}"
        python311 -m venv "${VENV_DIR}"
    else
        log "Virtual environment already present"
    fi

    if "$(venv_py)" -c "import torch, torchvision, grpc" >/dev/null 2>&1; then
        log "Dependencies already installed"
        return
    fi

    log "Upgrading pip …"
    "$(venv_py)" -m pip install --upgrade pip setuptools wheel --quiet

    if ! "$(venv_py)" -c "import torch" >/dev/null 2>&1; then
        local idx_url
        idx_url=$(cuda_index_url)
        log "Installing PyTorch from ${idx_url} …"
        "$(venv_py)" -m pip install torch torchvision --index-url "$idx_url" --quiet
    else
        log "PyTorch already installed"
    fi

    log "Installing project requirements …"
    "$(venv_py)" -m pip install -r "${REPO_DIR}/requirements.txt" --quiet
}

# ── Step 7: Proto stubs ───────────────────────────────────────────────────────

ensure_proto() {
    local pb2="${REPO_DIR}/proto/trainer_pb2.py"
    local pb2grpc="${REPO_DIR}/proto/trainer_pb2_grpc.py"
    if [ -f "$pb2" ] && [ -f "$pb2grpc" ]; then
        log "Proto stubs already generated"; return
    fi
    log "Generating gRPC proto stubs …"
    local proto_dir="${REPO_DIR}/proto"
    "$(venv_py)" -m grpc_tools.protoc \
        "--proto_path=${proto_dir}" \
        "--python_out=${proto_dir}" \
        "--grpc_python_out=${proto_dir}" \
        "${proto_dir}/trainer.proto"
    sed -i 's/^import trainer_pb2/from . import trainer_pb2/' "$pb2grpc"
    touch "${proto_dir}/__init__.py"
    log "Proto stubs generated"
}

# ── Step 8: Dataset ───────────────────────────────────────────────────────────

ensure_dataset() {
    if [ "${SKIP_DATASET}" = "1" ]; then
        log "Skipping dataset download (SKIP_DATASET=1)"; return
    fi
    log "Checking dataset …"
    "$(venv_py)" "${REPO_DIR}/dataset.py" \
        || log "Dataset download will be retried by the worker on first run."
}

# ── Step 9: Launch worker ─────────────────────────────────────────────────────

launch_worker() {
    log "Hardware detected:"
    "$(venv_py)" "${REPO_DIR}/hardware_probe.py" 2>/dev/null \
        | grep -E '"score"|"type"|"name"' | head -10 || true

    log "Starting worker → leader=${LEADER_HOST}:${LEADER_PORT}"
    exec "$(venv_py)" "${REPO_DIR}/worker.py" \
        --leader "${LEADER_HOST}" \
        --port   "${LEADER_PORT}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    log "=== Distributed ResNet Worker Bootstrap (Windows / Git Bash) ==="
    ensure_winget
    ensure_python311
    ensure_tailscale
    ensure_repo
    cd "${REPO_DIR}"
    ensure_venv
    ensure_proto
    ensure_dataset
    launch_worker
}

main "$@"
