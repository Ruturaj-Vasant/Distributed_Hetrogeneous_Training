# worker_windows.ps1  —  One-shot bootstrap + launch for a Windows worker node
#
# Usage (run in PowerShell as Administrator for first-time installs):
#   $env:LEADER_HOST="leader-macbook-pro.taila5426e.ts.net"; .\scripts\worker_windows.ps1
#
# What this script does:
#   1.  Ensures winget is available
#   2.  Installs Git (if missing)
#   3.  Installs Python 3.11 (if missing)
#   4.  Installs / authenticates Tailscale (if missing)
#   5.  Clones the repo (or pulls latest if already cloned)
#   6.  Creates a Python virtual environment
#   7.  Installs all Python dependencies
#       - Detects NVIDIA GPU and installs CUDA-enabled PyTorch automatically
#       - Falls back to CPU-only PyTorch if no GPU is found
#   8.  Generates gRPC proto stubs
#   9.  Downloads Tiny ImageNet-200 dataset (~236 MB, skipped if cached)
#  10.  Runs the worker
#
# Environment variables (all optional):
#   LEADER_HOST  — leader Tailscale DNS or IP  (default: leader-macbook-pro.taila5426e.ts.net)
#   LEADER_PORT  — gRPC port                   (default: 50051)
#   REPO_URL     — git clone URL
#   REPO_DIR     — local clone path             (default: $HOME\distributed-resnet)
#   SKIP_DATASET — set to "1" to skip dataset download

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$LeaderHost  = if ($env:LEADER_HOST)  { $env:LEADER_HOST  } else { "leader-macbook-pro.taila5426e.ts.net" }
$LeaderPort  = if ($env:LEADER_PORT)  { $env:LEADER_PORT  } else { "50051" }
$RepoUrl     = if ($env:REPO_URL)     { $env:REPO_URL     } else { "https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git" }
$RepoDir     = if ($env:REPO_DIR)     { $env:REPO_DIR     } else { Join-Path $HOME "distributed-resnet" }
$VenvDir     = Join-Path $RepoDir ".venv"
$SkipDataset = if ($env:SKIP_DATASET) { $env:SKIP_DATASET } else { "0" }

# ── Logging ───────────────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Msg)
    Write-Host "[worker:win] $Msg" -ForegroundColor Cyan
}

function Write-Err {
    param([string]$Msg)
    Write-Host "[worker:win] ERROR: $Msg" -ForegroundColor Red
    exit 1
}

function Refresh-Path {
    $parts = @(
        [Environment]::GetEnvironmentVariable("Path", "Machine"),
        [Environment]::GetEnvironmentVariable("Path", "User")
    ) | Where-Object { $_ }
    $env:Path = $parts -join ";"
}

function Get-Cmd {
    param([string]$Name)
    $c = Get-Command $Name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    return $null
}

function Invoke-Ok {
    param([string]$Exe, [string[]]$Arguments)
    $resolved = if (Test-Path $Exe) { $Exe } else { Get-Cmd $Exe }
    if (-not $resolved) { throw "Not found: $Exe" }
    & "$resolved" @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "Exit $LASTEXITCODE : $resolved $($Arguments -join ' ')" }
}

# ── Step 1: winget ────────────────────────────────────────────────────────────

function Ensure-Winget {
    if (Get-Cmd "winget.exe") { Write-Step "winget already present"; return }
    Write-Step "Installing winget (Microsoft App Installer) …"
    $pkg = Join-Path $env:TEMP "AppInstaller.msixbundle"
    Invoke-WebRequest -Uri "https://aka.ms/getwinget" -OutFile $pkg
    Add-AppxPackage -Path $pkg
    Refresh-Path
    if (-not (Get-Cmd "winget.exe")) { Write-Err "winget install failed. Install App Installer from the Microsoft Store, then re-run." }
}

function Ensure-WingetPkg {
    param([string]$Cmd, [string]$PkgId)
    if (Get-Cmd $Cmd) { Write-Step "$Cmd already present"; return }
    Write-Step "Installing $PkgId via winget …"
    Invoke-Ok "winget.exe" @("install","--id",$PkgId,"--exact","--accept-source-agreements","--accept-package-agreements","--silent")
    Refresh-Path
}

# ── Step 2: Git ───────────────────────────────────────────────────────────────
# (handled via Ensure-WingetPkg in Main)

# ── Step 3: Python 3.11 ───────────────────────────────────────────────────────

function Find-Python311 {
    $candidates = @(
        (Get-Cmd "py.exe"),
        "$env:SystemRoot\py.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    foreach ($c in $candidates) {
        $launcherArgs = if ($c -match "py\.exe$") { @("-3.11") } else { @() }
        $probeArgs = $launcherArgs + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        try {
            $ver = (& "$c" @probeArgs 2>&1 | Select-Object -First 1)
            if ($LASTEXITCODE -eq 0 -and $ver -eq "3.11") {
                return [pscustomobject]@{ Exe = $c; Extra = $launcherArgs }
            }
        } catch {}
    }
    return $null
}

function Ensure-Python311 {
    $py = Find-Python311
    if ($py) { Write-Step "Python 3.11 found: $($py.Exe)"; return $py }

    Write-Step "Installing Python 3.11 via winget …"
    Invoke-Ok "winget.exe" @("install","--id","Python.Python.3.11","--exact",
        "--accept-source-agreements","--accept-package-agreements","--silent")
    Refresh-Path
    Start-Sleep -Seconds 3

    $py = Find-Python311
    if (-not $py) { Write-Err "Python 3.11 not found after install. Open a new PowerShell and re-run." }
    Write-Step "Python 3.11 ready: $($py.Exe)"
    return $py
}

# ── Step 4: Tailscale ─────────────────────────────────────────────────────────

function Find-Tailscale {
    $candidates = @(
        (Get-Cmd "tailscale.exe"),
        "$env:ProgramFiles\Tailscale\tailscale.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
    if ($candidates) { return $candidates[0] }
    return $null
}

function Test-TailscaleRunning {
    param([string]$Exe)
    try {
        $s = (& "$Exe" status --json 2>&1 | Out-String)
        return $s -match '"BackendState"\s*:\s*"Running"'
    } catch { return $false }
}

function Ensure-Tailscale {
    $ts = Find-Tailscale
    if (-not $ts) {
        Write-Step "Installing Tailscale via winget …"
        Invoke-Ok "winget.exe" @("install","--id","Tailscale.Tailscale","--exact",
            "--accept-source-agreements","--accept-package-agreements","--silent")
        Refresh-Path
        $ts = Find-Tailscale
    } else {
        Write-Step "Tailscale already present"
    }
    if (-not $ts) { Write-Err "Tailscale not found after install." }

    $svc = Get-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne "Running") {
        Write-Step "Starting Tailscale service …"
        Start-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    }

    if (Test-TailscaleRunning $ts) { Write-Step "Tailscale is authenticated"; return }

    Write-Step "Authenticate Tailscale — a browser window will open (or copy the URL below):"
    $out = (& "$ts" up --timeout=1s 2>&1 | Out-String)
    Write-Host $out
    $url = [regex]::Match($out, "https://\S+").Value
    if ($url) { Start-Process $url } else { & "$ts" up }

    Write-Step "Waiting for Tailscale authentication …"
    while (-not (Test-TailscaleRunning $ts)) { Start-Sleep -Seconds 5 }
    Write-Step "Tailscale is authenticated"
}

# ── Step 5: Clone / update repo ───────────────────────────────────────────────

function Ensure-Repo {
    if (Test-Path (Join-Path $RepoDir ".git")) {
        Write-Step "Repo already present at $RepoDir — pulling latest"
        Invoke-Ok "git.exe" @("-C", $RepoDir, "pull", "--ff-only")
    } else {
        Write-Step "Cloning $RepoUrl → $RepoDir"
        Invoke-Ok "git.exe" @("clone", $RepoUrl, $RepoDir)
    }
}

# ── Step 6 & 7: Venv + dependencies ──────────────────────────────────────────

function Detect-CudaVersion {
    $nvidiaSmi = Get-Cmd "nvidia-smi.exe"
    if (-not $nvidiaSmi) { return $null }
    try {
        $out   = (& "$nvidiaSmi" 2>&1 | Out-String)
        $match = [regex]::Match($out, "CUDA Version:\s*([\d\.]+)")
        if ($match.Success) { return $match.Groups[1].Value }
    } catch {}
    return $null
}

function Get-TorchIndexUrl {
    $cuda = Detect-CudaVersion
    if (-not $cuda) {
        Write-Step "No NVIDIA GPU detected — installing CPU-only PyTorch"
        return "https://download.pytorch.org/whl/cpu"
    }
    $major = [int]($cuda -split "\.")[0]
    $minor = [int]($cuda -split "\.")[1]
    Write-Step "CUDA $cuda detected"
    if     ($major -ge 12 -and $minor -ge 4) { return "https://download.pytorch.org/whl/cu124" }
    elseif ($major -ge 12 -and $minor -ge 1) { return "https://download.pytorch.org/whl/cu121" }
    elseif ($major -ge 11 -and $minor -ge 8) { return "https://download.pytorch.org/whl/cu118" }
    else {
        Write-Step "CUDA $cuda is older than 11.8 — falling back to CPU-only PyTorch"
        return "https://download.pytorch.org/whl/cpu"
    }
}

function Ensure-Venv {
    param([object]$PySpec)

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (-not (Test-Path $VenvPython)) {
        Write-Step "Creating virtual environment at $VenvDir"
        $venvArgs = $PySpec.Extra + @("-m", "venv", $VenvDir)
        & $PySpec.Exe @venvArgs 2>&1 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { Write-Err "venv creation failed" }
    } else {
        Write-Step "Virtual environment already present"
    }

    # Check if deps are already installed
    $probe = & "$VenvPython" -c "import torch, torchvision, grpc" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Dependencies already installed"
        return $VenvPython
    }

    Write-Step "Upgrading pip …"
    Invoke-Ok $VenvPython @("-m","pip","install","--upgrade","pip","setuptools","wheel","--quiet")

    # PyTorch — choose CUDA or CPU build based on detected GPU
    $probe2 = & "$VenvPython" -c "import torch" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $indexUrl = Get-TorchIndexUrl
        Write-Step "Installing PyTorch from $indexUrl …"
        Invoke-Ok $VenvPython @("-m","pip","install","torch","torchvision",
            "--index-url",$indexUrl,"--quiet")
    } else {
        Write-Step "PyTorch already installed"
    }

    Write-Step "Installing project requirements …"
    Invoke-Ok $VenvPython @("-m","pip","install","-r",(Join-Path $RepoDir "requirements.txt"),"--quiet")

    return $VenvPython
}

# ── Step 8: Proto stubs ───────────────────────────────────────────────────────

function Ensure-Proto {
    param([string]$VenvPython)

    $pb2     = Join-Path $RepoDir "proto\trainer_pb2.py"
    $pb2grpc = Join-Path $RepoDir "proto\trainer_pb2_grpc.py"

    if ((Test-Path $pb2) -and (Test-Path $pb2grpc)) {
        Write-Step "Proto stubs already generated"
        return
    }

    Write-Step "Generating gRPC proto stubs …"
    $protoDir = Join-Path $RepoDir "proto"
    Invoke-Ok $VenvPython @("-m","grpc_tools.protoc",
        "--proto_path=$protoDir",
        "--python_out=$protoDir",
        "--grpc_python_out=$protoDir",
        (Join-Path $protoDir "trainer.proto"))

    # Fix absolute import → relative
    $grpcFile = Join-Path $protoDir "trainer_pb2_grpc.py"
    $content  = Get-Content $grpcFile -Raw
    $fixed    = $content -replace "^import trainer_pb2", "from . import trainer_pb2"
    Set-Content $grpcFile $fixed

    # Ensure __init__.py
    $init = Join-Path $protoDir "__init__.py"
    if (-not (Test-Path $init)) { New-Item -ItemType File $init | Out-Null }

    Write-Step "Proto stubs generated"
}

# ── Step 9: Dataset ───────────────────────────────────────────────────────────

function Ensure-Dataset {
    param([string]$VenvPython)
    if ($SkipDataset -eq "1") {
        Write-Step "Skipping dataset download (SKIP_DATASET=1)"
        return
    }
    Write-Step "Checking dataset …"
    & "$VenvPython" (Join-Path $RepoDir "dataset.py")
    if ($LASTEXITCODE -ne 0) { Write-Step "Dataset download will be retried by the worker on first run." }
}

# ── Step 10: Launch ───────────────────────────────────────────────────────────

function Start-Worker {
    param([string]$VenvPython)
    Write-Step "Hardware detected:"
    & "$VenvPython" (Join-Path $RepoDir "hardware_probe.py") 2>$null | Select-String '"score"|"type"|"name"'

    Write-Step "Starting worker → leader=${LeaderHost}:${LeaderPort}"
    & "$VenvPython" (Join-Path $RepoDir "worker.py") --leader $LeaderHost --port $LeaderPort
}

# ── Main ──────────────────────────────────────────────────────────────────────

function Main {
    Write-Step "=== Distributed ResNet Worker Bootstrap (Windows) ==="

    Ensure-Winget
    Ensure-WingetPkg "git.exe"  "Git.Git"

    $py         = Ensure-Python311
    Ensure-Tailscale
    Ensure-Repo

    Set-Location $RepoDir

    $venvPython = Ensure-Venv $py
    Ensure-Proto $venvPython
    Ensure-Dataset $venvPython
    Start-Worker $venvPython
}

Main
