# worker_windows.ps1 - Bootstrap and launch a Windows worker node.
#
# Typical usage from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\worker_windows.ps1
#
# Useful options:
#   -LeaderHost <host>     Leader Tailscale DNS name or IP.
#   -LeaderPort <port>     Leader gRPC port.
#   -SkipDataset           Do not pre-download the dataset.
#   -SkipTailscale         Do not install/authenticate Tailscale.
#   -DryRun                Tell dtrain-worker to use synthetic data.
#   -SetupOnly             Install/check dependencies, then exit before launching dtrain-worker.
#   -NoInstall             Fail with instructions instead of installing missing tools.
#
# Environment variable equivalents are also supported:
#   LEADER_HOST, LEADER_PORT, REPO_URL, REPO_DIR, DATASET, PRELOAD_DATASETS, CACHE_DIR,
#   CONNECT_RETRIES,
#   SKIP_DATASET=1, SKIP_TAILSCALE=1, DRY_RUN=1, WORKER_SETUP_ONLY=1,
#   WORKER_NO_INSTALL=1, UPDATE_REPO=1, WORKER_NO_BROWSER=1

[CmdletBinding()]
param(
    [string]$LeaderHost,
    [int]$LeaderPort = 0,
    [string]$RepoUrl,
    [string]$RepoDir,
    [string]$Dataset,
    [string]$PreloadDatasets,
    [string]$CacheDir,
    [int]$ConnectRetries = 0,
    [switch]$SkipDataset,
    [switch]$SkipTailscale,
    [switch]$DryRun,
    [switch]$SetupOnly,
    [switch]$NoInstall,
    [switch]$UpdateRepo,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$WorkerArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-Truthy {
    param([string]$Value)
    return $Value -match '^(1|true|yes|y|on)$'
}

function First-Value {
    param([object[]]$Values)
    foreach ($value in $Values) {
        if ($null -ne $value -and "$value".Trim() -ne "") { return "$value" }
    }
    return $null
}

function Get-DatasetList {
    param(
        [string]$Raw,
        [string]$PrimaryDataset
    )

    $valid = @("tinyimagenet", "cifar10")
    $result = New-Object System.Collections.Generic.List[string]
    foreach ($item in ($Raw -split ",")) {
        $name = $item.Trim().ToLowerInvariant()
        if (-not $name) { continue }
        if ($name -eq "all") {
            foreach ($datasetName in $valid) {
                if (-not $result.Contains($datasetName)) { [void]$result.Add($datasetName) }
            }
            continue
        }
        if ($name -notin $valid) {
            Write-Err "Unsupported dataset '$name'. Expected tinyimagenet, cifar10, or all."
        }
        if (-not $result.Contains($name)) { [void]$result.Add($name) }
    }

    if (-not $result.Contains($PrimaryDataset)) {
        $result.Insert(0, $PrimaryDataset)
    }
    return @($result.ToArray())
}

function Write-Step {
    param([string]$Msg)
    Write-Host "[worker:win] $Msg" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Msg)
    Write-Host "[worker:win] WARNING: $Msg" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Msg)
    Write-Host "[worker:win] ERROR: $Msg" -ForegroundColor Red
    exit 1
}

function Get-Cmd {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Refresh-Path {
    $paths = New-Object System.Collections.Generic.List[string]
    foreach ($scope in @("Machine", "User", "Process")) {
        $raw = [Environment]::GetEnvironmentVariable("Path", $scope)
        if (-not $raw) { continue }
        foreach ($part in ($raw -split ";")) {
            $trimmed = $part.Trim()
            if ($trimmed -and -not $paths.Contains($trimmed)) {
                [void]$paths.Add($trimmed)
            }
        }
    }
    $env:Path = $paths -join ";"
}

function Invoke-Ok {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [int[]]$OkCodes = @(0)
    )
    $resolved = if (Test-Path -LiteralPath $Exe) { $Exe } else { Get-Cmd $Exe }
    if (-not $resolved) { throw "Not found: $Exe" }

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & "$resolved" @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }

    if ($exitCode -notin $OkCodes) {
        throw "Exit $exitCode : $resolved $($Arguments -join ' ')"
    }
}

function Test-PythonImports {
    param(
        [string]$PythonExe,
        [string]$ImportCode
    )
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & "$PythonExe" -c $ImportCode *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Ensure-Winget {
    if (Get-Cmd "winget.exe") {
        Write-Step "winget already present"
        return
    }
    if ($NoInstallFlag) {
        Write-Err "winget is not installed. Install Microsoft App Installer, then re-run."
    }

    Write-Step "Installing winget (Microsoft App Installer)"
    $pkg = Join-Path $env:TEMP "Microsoft.DesktopAppInstaller.msixbundle"
    Invoke-WebRequest -Uri "https://aka.ms/getwinget" -OutFile $pkg
    Add-AppxPackage -Path $pkg
    Refresh-Path

    if (-not (Get-Cmd "winget.exe")) {
        Write-Err "winget install failed. Install App Installer from the Microsoft Store, then re-run."
    }
}

function Ensure-WingetPkg {
    param(
        [string]$Cmd,
        [string]$PkgId,
        [string]$FriendlyName
    )
    if (Get-Cmd $Cmd) {
        Write-Step "$FriendlyName already present"
        return
    }
    if ($NoInstallFlag) {
        Write-Err "$FriendlyName is missing. Install $PkgId or re-run without -NoInstall."
    }

    Ensure-Winget
    Write-Step "Installing $FriendlyName via winget"
    Invoke-Ok "winget.exe" @(
        "install", "--id", $PkgId, "--exact",
        "--accept-source-agreements", "--accept-package-agreements", "--silent"
    ) -OkCodes @(0, -1978335189)
    Refresh-Path
}

function Find-Python311 {
    $pyLaunchers = @(
        (Get-Cmd "py.exe"),
        (Join-Path $env:SystemRoot "py.exe"),
        (Join-Path $env:SystemRoot "System32\py.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

    foreach ($launcher in $pyLaunchers) {
        try {
            $ver = (& "$launcher" -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1 | Select-Object -First 1)
            if ("$ver".Trim() -eq "3.11") {
                return [pscustomobject]@{ Exe = $launcher; Extra = @("-3.11") }
            }
        } catch {}
    }

    $directPaths = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python\Python311\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python311\python.exe"),
        (Get-Cmd "python3.11.exe"),
        (Get-Cmd "python.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

    foreach ($candidate in $directPaths) {
        try {
            $ver = (& "$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1 | Select-Object -First 1)
            if ("$ver".Trim() -eq "3.11") {
                return [pscustomobject]@{ Exe = $candidate; Extra = @() }
            }
        } catch {}
    }
    return $null
}

function Ensure-Python311 {
    $py = Find-Python311
    if ($py) {
        Write-Step "Python 3.11 found: $($py.Exe)"
        return $py
    }

    if ($NoInstallFlag) {
        Write-Err "Python 3.11 is missing. Install Python.Python.3.11 or re-run without -NoInstall."
    }

    Ensure-Winget
    Write-Step "Installing Python 3.11 via winget"
    Invoke-Ok "winget.exe" @(
        "install", "--id", "Python.Python.3.11", "--exact",
        "--accept-source-agreements", "--accept-package-agreements", "--silent"
    ) -OkCodes @(0, -1978335189)
    Start-Sleep -Seconds 3
    Refresh-Path

    $py = Find-Python311
    if (-not $py) {
        Write-Err "Python 3.11 not found after install. Open a new PowerShell window and re-run."
    }
    Write-Step "Python 3.11 ready: $($py.Exe)"
    return $py
}

function Find-Tailscale {
    $candidates = @(
        (Get-Cmd "tailscale.exe"),
        (Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Tailscale\tailscale.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

    if (@($candidates).Count -gt 0) { return @($candidates)[0] }
    return $null
}

function Test-TailscaleRunning {
    param([string]$Exe)
    try {
        $status = (& "$Exe" status --json 2>&1 | Out-String)
        return $status -match '"BackendState"\s*:\s*"Running"'
    } catch {
        return $false
    }
}

function Ensure-Tailscale {
    if ($SkipTailscaleFlag) {
        Write-Step "Skipping Tailscale setup"
        return
    }

    $ts = Find-Tailscale
    if (-not $ts) {
        Ensure-WingetPkg "tailscale.exe" "Tailscale.Tailscale" "Tailscale"
        $ts = Find-Tailscale
    } else {
        Write-Step "Tailscale already present"
    }
    if (-not $ts) { Write-Err "Tailscale not found after install." }

    $svc = Get-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne "Running") {
        Write-Step "Starting Tailscale service"
        try { Start-Service -Name "Tailscale" } catch { Write-Warn "Could not start Tailscale service: $_" }
    }

    if (Test-TailscaleRunning $ts) {
        Write-Step "Tailscale is authenticated"
        return
    }

    Write-Step "Authenticating Tailscale"
    $out = (& "$ts" up --timeout=1s 2>&1 | Out-String)
    Write-Host $out

    $url = [regex]::Match($out, "https://\S+").Value
    if ($url -and -not (Test-Truthy $env:WORKER_NO_BROWSER)) {
        Start-Process $url
    } elseif ($url) {
        Write-Host "[worker:win] Open this URL to authenticate: $url"
    } else {
        & "$ts" up
    }

    Write-Step "Waiting for Tailscale authentication"
    while (-not (Test-TailscaleRunning $ts)) { Start-Sleep -Seconds 5 }
    Write-Step "Tailscale is authenticated"
}

function Ensure-Repo {
    if (Test-Path -LiteralPath (Join-Path $RepoDir ".git")) {
        Write-Step "Using repo at $RepoDir"
        if ($UpdateRepoFlag) {
            Ensure-WingetPkg "git.exe" "Git.Git" "Git"
            Write-Step "Pulling latest changes"
            Invoke-Ok "git.exe" @("-C", $RepoDir, "pull", "--ff-only")
        }
        return
    }

    if ((Test-Path -LiteralPath $RepoDir) -and -not (Test-Path -LiteralPath (Join-Path $RepoDir ".git"))) {
        Write-Err "REPO_DIR exists but is not a git checkout: $RepoDir"
    }

    Ensure-WingetPkg "git.exe" "Git.Git" "Git"
    Write-Step "Cloning $RepoUrl to $RepoDir"
    Invoke-Ok "git.exe" @("clone", $RepoUrl, $RepoDir)
}

function Detect-CudaVersion {
    $nvidiaSmi = Get-Cmd "nvidia-smi.exe"
    if (-not $nvidiaSmi) { return $null }
    try {
        $out = (& "$nvidiaSmi" 2>&1 | Out-String)
        $match = [regex]::Match($out, "CUDA Version:\s*([\d\.]+)")
        if ($match.Success) { return $match.Groups[1].Value }
    } catch {}
    return $null
}

function Get-TorchIndexUrl {
    $cuda = Detect-CudaVersion
    if (-not $cuda) {
        Write-Step "No NVIDIA GPU detected; installing CPU-only PyTorch"
        return "https://download.pytorch.org/whl/cpu"
    }

    $parts = $cuda -split "\."
    $major = [int]$parts[0]
    $minor = if ($parts.Count -gt 1) { [int]$parts[1] } else { 0 }
    Write-Step "CUDA $cuda detected"

    if ($major -gt 12 -or ($major -eq 12 -and $minor -ge 4)) { return "https://download.pytorch.org/whl/cu124" }
    if ($major -eq 12 -and $minor -ge 1) { return "https://download.pytorch.org/whl/cu121" }
    if ($major -eq 11 -and $minor -ge 8) { return "https://download.pytorch.org/whl/cu118" }

    Write-Step "CUDA $cuda is older than 11.8; falling back to CPU-only PyTorch"
    return "https://download.pytorch.org/whl/cpu"
}

function Repair-PipArtifacts {
    param([string]$VenvPython)

    $venvRoot = Split-Path -Parent (Split-Path -Parent $VenvPython)
    $sitePackages = Join-Path $venvRoot "Lib\site-packages"
    if (-not (Test-Path -LiteralPath $sitePackages)) { return }

    $artifacts = @(Get-ChildItem -LiteralPath $sitePackages -Force -Filter "~ip*" -ErrorAction SilentlyContinue)
    if ($artifacts.Count -eq 0) { return }

    Write-Step "Cleaning incomplete pip upgrade artifacts"
    foreach ($artifact in $artifacts) {
        try {
            Remove-Item -LiteralPath $artifact.FullName -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Warn "Could not remove $($artifact.FullName): $_"
        }
    }
}

function Ensure-Venv {
    param([object]$PySpec)

    $venvDir = Join-Path $RepoDir ".venv"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Step "Creating virtual environment at $venvDir"
        $venvArgs = @($PySpec.Extra) + @("-m", "venv", $venvDir)
        & $PySpec.Exe @venvArgs 2>&1 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { Write-Err "venv creation failed" }
    } else {
        Write-Step "Virtual environment already present"
    }

    Repair-PipArtifacts $venvPython

    $depsReady = Test-PythonImports $venvPython "import torch, torchvision, grpc, grpc_tools, psutil, numpy"
    if ($depsReady) {
        Write-Step "Dependencies already installed"
        return $venvPython
    }

    Write-Step "Upgrading pip"
    Invoke-Ok $venvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet")

    $torchReady = Test-PythonImports $venvPython "import torch, torchvision"
    if (-not $torchReady) {
        $indexUrl = Get-TorchIndexUrl
        Write-Step "Installing PyTorch from $indexUrl"
        Invoke-Ok $venvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", $indexUrl, "--quiet")
    } else {
        Write-Step "PyTorch already installed"
    }

    Write-Step "Installing package and dependencies"
    Invoke-Ok $venvPython @("-m", "pip", "install", "-e", $RepoDir, "--quiet")

    return $venvPython
}

function Ensure-Proto {
    param([string]$VenvPython)

    $protoDir = Join-Path $RepoDir "proto"
    $pb2 = Join-Path $protoDir "trainer_pb2.py"
    $pb2grpc = Join-Path $protoDir "trainer_pb2_grpc.py"

    if ((Test-Path -LiteralPath $pb2) -and (Test-Path -LiteralPath $pb2grpc)) {
        Write-Step "Proto stubs already generated"
        return
    }

    Write-Step "Generating gRPC proto stubs"
    Invoke-Ok $VenvPython @(
        "-m", "grpc_tools.protoc",
        "--proto_path=$protoDir",
        "--python_out=$protoDir",
        "--grpc_python_out=$protoDir",
        (Join-Path $protoDir "trainer.proto")
    )

    $content = Get-Content $pb2grpc -Raw
    $fixed = $content -replace "(?m)^import trainer_pb2", "from . import trainer_pb2"
    Set-Content -LiteralPath $pb2grpc -Value $fixed -Encoding UTF8

    $init = Join-Path $protoDir "__init__.py"
    if (-not (Test-Path -LiteralPath $init)) { New-Item -ItemType File -Path $init | Out-Null }
    Write-Step "Proto stubs generated"
}

function Ensure-Dataset {
    param([string]$VenvPython)

    if ($SkipDatasetFlag) {
        Write-Step "Skipping dataset download"
        return
    }

    foreach ($datasetName in $DatasetList) {
        Write-Step "Checking $datasetName dataset"
        $dsArgs = @("-c", "from trainer.data import ensure_any_dataset; ensure_any_dataset('$datasetName')")
        if ($CacheDir) { $dsArgs = @("-c", "from trainer.data import ensure_any_dataset; ensure_any_dataset('$datasetName', '$CacheDir')") }
        & "$VenvPython" @dsArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "$datasetName setup failed; dtrain-worker will retry if the leader requests it."
        }
    }
}

function Show-Hardware {
    param([string]$VenvPython)
    Write-Step "Hardware detected"
    try {
        & "$VenvPython" "-c" "from trainer.utils.hardware import probe_to_dict; import json; print(json.dumps(probe_to_dict(), indent=2))" 2>$null |
            Select-String '"score"|"type"|"name"|"cpu_cores"|"ram_gb"'
    } catch {
        Write-Warn "Hardware probe failed: $_"
    }
}

function Test-LeaderPort {
    if (-not (Get-Command Test-NetConnection -ErrorAction SilentlyContinue)) { return }
    Write-Step "Checking leader TCP reachability: ${LeaderHost}:${LeaderPort}"
    try {
        $ok = Test-NetConnection -ComputerName $LeaderHost -Port $LeaderPort -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($ok) {
            Write-Step "Leader port is reachable"
        } else {
            Write-Warn "Leader port is not reachable yet. dtrain-worker will keep retrying."
        }
    } catch {
        Write-Warn "Leader reachability check failed: $_"
    }
}

function Start-Worker {
    param([string]$VenvPython)

    Show-Hardware $VenvPython
    Test-LeaderPort

    $dtrain = Join-Path (Split-Path -Parent $VenvPython) "dtrain-worker.exe"
    if (-not (Test-Path -LiteralPath $dtrain)) { $dtrain = Join-Path (Split-Path -Parent $VenvPython) "dtrain-worker" }

    $launchArgs = @(
        "--leader", $LeaderHost,
        "--port", "$LeaderPort",
        "--dataset", $Dataset,
        "--preload-datasets", ($DatasetList -join ","),
        "--connect-retries", "$ConnectRetries"
    )
    if ($CacheDir) { $launchArgs += @("--cache-dir", $CacheDir) }
    if ($DryRunFlag) { $launchArgs += "--dry-run" }
    if ($WorkerArgs) { $launchArgs += $WorkerArgs }

    Write-Step "Starting worker -> leader=${LeaderHost}:${LeaderPort}"
    & "$dtrain" @launchArgs
    exit $LASTEXITCODE
}

$scriptPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
$scriptRoot = Split-Path -Parent $scriptPath
$repoBesideScript = Resolve-Path -LiteralPath (Join-Path $scriptRoot "..") -ErrorAction SilentlyContinue

$LeaderHost = First-Value @($LeaderHost, $env:LEADER_HOST, "leader-macbook-pro.taila5426e.ts.net")
$LeaderPort = if ($LeaderPort -gt 0) { $LeaderPort } elseif ($env:LEADER_PORT) { [int]$env:LEADER_PORT } else { 50051 }
$ConnectRetries = if ($ConnectRetries -gt 0) { $ConnectRetries } elseif ($env:CONNECT_RETRIES) { [int]$env:CONNECT_RETRIES } else { 2147483647 }
$RepoUrl = First-Value @($RepoUrl, $env:REPO_URL, "https://github.com/Ruturaj-Vasant/Distributed_Hetrogeneous_Training.git")
$Dataset = First-Value @($Dataset, $env:DATASET, "tinyimagenet")
$PreloadDatasets = First-Value @($PreloadDatasets, $env:PRELOAD_DATASETS, "tinyimagenet,cifar10")
$CacheDir = First-Value @($CacheDir, $env:CACHE_DIR)

if (-not $RepoDir) {
    if ($env:REPO_DIR) {
        $RepoDir = $env:REPO_DIR
    } elseif ($repoBesideScript -and (Test-Path -LiteralPath (Join-Path $repoBesideScript ".git"))) {
        $RepoDir = $repoBesideScript.Path
    } else {
        $RepoDir = Join-Path $HOME "distributed-resnet"
    }
}

$RepoDir = [System.IO.Path]::GetFullPath($RepoDir)

$SkipDatasetFlag = [bool]$SkipDataset -or (Test-Truthy $env:SKIP_DATASET)
$SkipTailscaleFlag = [bool]$SkipTailscale -or (Test-Truthy $env:SKIP_TAILSCALE)
$DryRunFlag = [bool]$DryRun -or (Test-Truthy $env:DRY_RUN)
$SetupOnlyFlag = [bool]$SetupOnly -or (Test-Truthy $env:WORKER_SETUP_ONLY)
$NoInstallFlag = [bool]$NoInstall -or (Test-Truthy $env:WORKER_NO_INSTALL)
$UpdateRepoFlag = [bool]$UpdateRepo -or (Test-Truthy $env:UPDATE_REPO)

if ($WorkerArgs -and $WorkerArgs.Count -gt 0 -and $WorkerArgs[0].ToLowerInvariant() -eq "run") {
    $WorkerArgs = if ($WorkerArgs.Count -gt 1) { $WorkerArgs[1..($WorkerArgs.Count - 1)] } else { @() }
}

if ($Dataset -notin @("tinyimagenet", "cifar10")) {
    Write-Err "Unsupported dataset '$Dataset'. Expected tinyimagenet or cifar10."
}
$DatasetList = Get-DatasetList $PreloadDatasets $Dataset

function Main {
    Write-Step "=== Distributed ResNet Worker Bootstrap (Windows) ==="
    Write-Step "Repo: $RepoDir"
    Write-Step "Leader: ${LeaderHost}:${LeaderPort}"

    Ensure-Repo
    Set-Location $RepoDir

    $py = Ensure-Python311
    Ensure-Tailscale

    $venvPython = Ensure-Venv $py
    Ensure-Proto $venvPython
    Ensure-Dataset $venvPython

    if ($SetupOnlyFlag) {
        Show-Hardware $venvPython
        Test-LeaderPort
        Write-Step "Setup complete; not starting worker because SetupOnly is enabled"
        return
    }

    Start-Worker $venvPython
}

Main
