#!/usr/bin/env bash
# worker_macos.sh  —  One-shot bootstrap + launch for a Mac worker node
#
# Usage:
#   LEADER_HOST=leader-macbook-pro.taila5426e.ts.net bash worker_macos.sh
#
# What this script does:
#   1. Installs Xcode Command Line Tools (if missing)
#   2. Installs Homebrew (if missing)
#   3. Installs Python 3.11 via Homebrew (if missing)
#   4. Installs / authenticates Tailscale (if missing)
#   5. Clones the repo (or pulls latest if already cloned)
#   6. Creates a Python virtual environment
#   7. Installs all Python dependencies (including PyTorch with MPS support)
#   8. Generates gRPC proto stubs
#   9. Downloads Tiny ImageNet-200 dataset (~236 MB, skipped if cached)
#  10. Runs the worker (connects to leader, waits for training to start)
#
# Environment variables (all optional):
#   LEADER_HOST  — leader Tailscale DNS or IP  (default: leader-macbook-pro.taila5426e.ts.net)
#   LEADER_PORT  — gRPC port                   (default: 50051)
#   REPO_URL     — git clone URL               (default: HTTPS GitHub URL)
#   REPO_DIR     — local clone directory        (default: ~/distributed-resnet)
#   SKIP_DATASET — set to 1 to skip dataset download (worker downloads on first run)

set -euo pipefail

LEADER_HOST="${LEADER_HOST:-leader-macbook-pro.taila5426e.ts.net}"
LEADER_PORT="${LEADER_PORT:-50051}"
REPO_URL="${REPO_URL:-https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git}"
REPO_DIR="${REPO_DIR:-${HOME}/distributed-resnet}"
VENV_DIR="${REPO_DIR}/.venv"
SKIP_DATASET="${SKIP_DATASET:-0}"

# ── Logging ───────────────────────────────────────────────────────────────────

log() { printf '\033[1;36m[worker:mac]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[worker:mac] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# ── Step 1: Xcode Command Line Tools ─────────────────────────────────────────

ensure_xcode_clt() {
  if xcode-select -p >/dev/null 2>&1; then
    log "Xcode Command Line Tools already present"
    return
  fi
  log "Installing Xcode Command Line Tools — approve the macOS popup, then wait …"
  xcode-select --install >/dev/null 2>&1 || true
  until xcode-select -p >/dev/null 2>&1; do sleep 10; done
  log "Xcode Command Line Tools installed"
}

# ── Step 2: Homebrew ──────────────────────────────────────────────────────────

load_brew() {
  have brew && return
  [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)" && return
  [ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"    && return
}

ensure_homebrew() {
  load_brew
  if have brew; then log "Homebrew already present"; return; fi
  log "Installing Homebrew …"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_brew
  have brew || err "Homebrew installed but not on PATH — open a new terminal and re-run."
}

ensure_brew_pkg() {
  local cmd="$1" pkg="${2:-$1}"
  if have "${cmd}"; then log "${cmd} already present"; return; fi
  log "Installing ${pkg} via Homebrew"
  brew install "${pkg}"
}

# ── Step 3: Python 3.11 ───────────────────────────────────────────────────────

python311() {
  have python3.11 && { command -v python3.11; return; }
  local p
  p="$(brew --prefix python@3.11 2>/dev/null || true)"
  [ -n "${p}" ] && [ -x "${p}/bin/python3.11" ] && { printf '%s\n' "${p}/bin/python3.11"; return; }
  return 1
}

ensure_python311() {
  if python311 >/dev/null 2>&1; then
    log "Python 3.11 already present ($(python311))"
    return
  fi
  log "Installing Python 3.11 via Homebrew"
  brew install python@3.11
  python311 >/dev/null 2>&1 || err "Python 3.11 not found after install."
}

# ── Step 4: Tailscale ─────────────────────────────────────────────────────────

tailscale_bin() {
  have tailscale && { command -v tailscale; return; }
  local p
  p="$(brew --prefix tailscale 2>/dev/null || true)"
  [ -n "${p}" ] && [ -x "${p}/bin/tailscale" ] && { printf '%s\n' "${p}/bin/tailscale"; return; }
  return 1
}

tailscale_running() {
  local ts; ts="$(tailscale_bin 2>/dev/null || true)"
  [ -n "${ts}" ] || return 1
  "${ts}" status --json 2>/dev/null | grep -q '"BackendState"[[:space:]]*:[[:space:]]*"Running"'
}

ensure_tailscale() {
  local installed_now=0
  if tailscale_bin >/dev/null 2>&1; then
    log "Tailscale CLI already present"
  else
    log "Installing Tailscale via Homebrew"
    brew install tailscale
    installed_now=1
  fi

  brew services start tailscale >/dev/null 2>&1 || true

  if tailscale_running; then log "Tailscale is authenticated"; return; fi

  log "Authenticate Tailscale — a browser window will open (or copy the URL below):"
  local out; out="$("$(tailscale_bin)" up --timeout=1s 2>&1 || true)"
  printf '%s\n' "${out}"
  local url; url="$(printf '%s\n' "${out}" | grep -Eo 'https://[^[:space:]]+' | head -1 || true)"
  [ -n "${url}" ] && open "${url}" >/dev/null 2>&1 || "$(tailscale_bin)" up || true

  log "Waiting for Tailscale authentication …"
  until tailscale_running; do sleep 5; done
  log "Tailscale is authenticated"
}

# ── Step 5: Clone / update repo ───────────────────────────────────────────────

ensure_repo() {
  if [ -d "${REPO_DIR}/.git" ]; then
    log "Repo already cloned at ${REPO_DIR} — pulling latest"
    git -C "${REPO_DIR}" pull --ff-only
  else
    log "Cloning ${REPO_URL} → ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
}

# ── Step 6 & 7: Virtual environment + dependencies ────────────────────────────

venv_ready() {
  [ -x "${VENV_DIR}/bin/python" ] || return 1
  "${VENV_DIR}/bin/python" -c "import torch, torchvision, grpc" >/dev/null 2>&1
}

ensure_venv() {
  local py; py="$(python311)"
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${py}" -m venv "${VENV_DIR}"
  else
    log "Virtual environment already present"
  fi

  if venv_ready; then
    log "Dependencies already installed — skipping (set SKIP_VENV_CHECK=1 to force reinstall)"
    return
  fi

  log "Installing Python dependencies …"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip --quiet

  # PyTorch for macOS — pip's default build includes MPS support
  if ! "${VENV_DIR}/bin/python" -c "import torch" >/dev/null 2>&1; then
    log "Installing PyTorch (MPS-enabled macOS build) …"
    "${VENV_DIR}/bin/python" -m pip install torch torchvision --quiet
  else
    log "PyTorch already installed"
  fi

  log "Installing package and dependencies …"
  "${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --quiet
}

# ── Step 8: Proto stubs ───────────────────────────────────────────────────────

ensure_proto() {
  if [ -f "${REPO_DIR}/proto/trainer_pb2.py" ] && [ -f "${REPO_DIR}/proto/trainer_pb2_grpc.py" ]; then
    log "Proto stubs already generated"
    return
  fi
  log "Generating gRPC proto stubs …"
  PYTHON="${VENV_DIR}/bin/python" bash "${REPO_DIR}/scripts/generate_proto.sh"
}

# ── Step 9: Dataset ───────────────────────────────────────────────────────────

ensure_dataset_download() {
  if [ "${SKIP_DATASET}" = "1" ]; then
    log "Skipping dataset download (SKIP_DATASET=1) — worker will download on first run"
    return
  fi
  log "Checking dataset …"
  "${VENV_DIR}/bin/python" -c "from trainer.data import ensure_any_dataset; ensure_any_dataset('tinyimagenet')"
}

# ── Step 10: Launch worker ────────────────────────────────────────────────────

launch_worker() {
  log "Starting worker → leader=${LEADER_HOST}:${LEADER_PORT}"
  exec "${VENV_DIR}/bin/dtrain-worker" \
    --leader "${LEADER_HOST}" \
    --port   "${LEADER_PORT}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  log "=== Distributed ResNet Worker Bootstrap (macOS) ==="
  ensure_xcode_clt
  ensure_homebrew
  ensure_brew_pkg curl curl
  ensure_brew_pkg git  git
  ensure_python311
  ensure_tailscale
  ensure_repo
  cd "${REPO_DIR}"
  ensure_venv
  ensure_proto
  ensure_dataset_download
  launch_worker
}

main "$@"
