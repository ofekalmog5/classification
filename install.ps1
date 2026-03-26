#Requires -Version 5.1
<#
.SYNOPSIS
    Full installer for Classification Web App.
    Installs Python 3.11, creates a venv, installs all packages,
    installs Node.js, and builds the frontend.

.PARAMETER CPU
    Force CPU-only PyTorch (skip CUDA even if an NVIDIA GPU is found).

.PARAMETER SkipTorch
    Skip PyTorch installation entirely (useful if already installed in the venv).

.PARAMETER SkipSam
    Skip segment-geospatial / SAM installation (saves ~2 GB but disables
    road/building/vegetation extraction features).

.PARAMETER SkipFrontend
    Skip the Node.js / frontend build step.
#>
param(
    [switch]$CPU,
    [switch]$SkipTorch,
    [switch]$SkipSam,
    [switch]$SkipFrontend
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colours / helpers ──────────────────────────────────────────────────────────
function Write-Step   { param($n,$total,$msg) Write-Host "`n[$n/$total] $msg" -ForegroundColor Cyan }
function Write-OK     { param($msg) Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Info   { param($msg) Write-Host "      $msg" -ForegroundColor Gray }
function Write-Warn   { param($msg) Write-Host " WARN $msg" -ForegroundColor Yellow }
function Write-Fail   { param($msg) Write-Host " FAIL $msg" -ForegroundColor Red }

function Exit-WithError {
    param($msg)
    Write-Fail $msg
    exit 1
}

# ── Paths ──────────────────────────────────────────────────────────────────────
$ROOT       = $PSScriptRoot          # project root (where install.ps1 lives)
$VENV       = Join-Path $ROOT ".venv"
$VENV_PY    = Join-Path $VENV "Scripts\python.exe"
$VENV_PIP   = Join-Path $VENV "Scripts\pip.exe"
$BACKEND_REQ = Join-Path $ROOT "backend\requirements.txt"
$GPU_REQ     = Join-Path $ROOT "backend\requirements-gpu.txt"
$WEBAPP_DIR  = Join-Path $ROOT "web_app"
$DIST_INDEX  = Join-Path $WEBAPP_DIR "dist\index.html"
$TEMP_DIR    = $env:TEMP

$TOTAL_STEPS = 6

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Classification Web App - Installer" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Root : $ROOT"
Write-Host "  Venv : $VENV"

# ==============================================================================
# STEP 1 — Python 3.11
# ==============================================================================
Write-Step 1 $TOTAL_STEPS "Checking Python 3.11..."

function Find-Python311 {
    # Check PATH candidates
    $candidates = @("python", "python3", "py")
    foreach ($c in $candidates) {
        try {
            $v = & $c --version 2>&1
            if ($v -match "Python 3\.11") { return $c }
        } catch {}
    }
    # Check py launcher with explicit version
    try {
        $v = & py -3.11 --version 2>&1
        if ($v -match "Python 3\.11") { return "py -3.11" }
    } catch {}
    # Check common install locations
    $locations = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Python\pythoncore-3.11-64\python.exe",
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files (x86)\Python311\python.exe"
    )
    foreach ($p in $locations) {
        if (Test-Path $p) {
            try {
                $v = & $p --version 2>&1
                if ($v -match "Python 3\.11") { return $p }
            } catch {}
        }
    }
    return $null
}

$SYSTEM_PY = Find-Python311

if ($SYSTEM_PY) {
    $ver = & $SYSTEM_PY --version 2>&1
    Write-OK "Found: $ver  ($SYSTEM_PY)"
} else {
    Write-Warn "Python 3.11 not found. Downloading Python 3.11.9..."

    $pyUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $pyInstaller = Join-Path $TEMP_DIR "python-3.11.9-amd64.exe"

    Write-Info "Downloading from: $pyUrl"
    Write-Info "Saving to       : $pyInstaller"

    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($pyUrl, $pyInstaller)
    } catch {
        Exit-WithError "Download failed: $_`nCheck your internet connection."
    }

    Write-Info "Installing Python 3.11.9 (silent, for current user)..."
    $args = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_test=0",
        "Include_launcher=1",
        "InstallLauncherAllUsers=0"
    )
    $proc = Start-Process -FilePath $pyInstaller -ArgumentList $args -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Exit-WithError "Python installer exited with code $($proc.ExitCode)."
    }

    # Refresh PATH for this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" `
              + [System.Environment]::GetEnvironmentVariable("Path","Machine")

    # Try again
    $SYSTEM_PY = Find-Python311
    if (-not $SYSTEM_PY) {
        Exit-WithError "Python 3.11 still not found after install. Try restarting and re-running."
    }
    $ver = & $SYSTEM_PY --version 2>&1
    Write-OK "Installed: $ver"
}

# ==============================================================================
# STEP 2 — Virtual environment
# ==============================================================================
Write-Step 2 $TOTAL_STEPS "Setting up virtual environment..."

if (Test-Path $VENV_PY) {
    $venvVer = & $VENV_PY --version 2>&1
    Write-OK "Existing venv OK: $venvVer"
} else {
    Write-Info "Creating .venv at: $VENV"
    if ($SYSTEM_PY -like "py *") {
        # py launcher form: "py -3.11"
        $pyArgs = $SYSTEM_PY.Substring(3).Trim()  # "-3.11"
        & py $pyArgs -m venv $VENV
    } else {
        & $SYSTEM_PY -m venv $VENV
    }
    if (-not (Test-Path $VENV_PY)) {
        Exit-WithError "venv creation failed. $VENV_PY does not exist."
    }
    Write-OK "Virtual environment created."
}

# Upgrade pip + wheel inside the venv
Write-Info "Upgrading pip / setuptools / wheel..."
& $VENV_PY -m pip install --upgrade pip setuptools wheel --quiet
Write-OK "pip up to date."

# ==============================================================================
# STEP 3 — PyTorch
# ==============================================================================
Write-Step 3 $TOTAL_STEPS "Installing PyTorch..."

if ($SkipTorch) {
    Write-Warn "Skipping PyTorch (--SkipTorch flag set)."
} else {
    # Check if already installed
    $torchInstalled = & $VENV_PY -c "import torch; print(torch.__version__)" 2>&1
    if ($torchInstalled -notmatch "error|Error|No module") {
        Write-OK "PyTorch already installed: $torchInstalled"
    } else {
        # Detect CUDA
        $hasCuda = $false
        $cudaMajor = 0
        if (-not $CPU) {
            try {
                $smi = & "nvidia-smi" "--query-gpu=name" "--format=csv,noheader" 2>&1
                if ($smi -notmatch "error|not found|is not recognized|failed") {
                    $hasCuda = $true
                    # Parse CUDA version
                    $smiOut = & "nvidia-smi" 2>&1 | Out-String
                    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
                        $cudaMajor = [int]$Matches[1]
                        Write-Info "NVIDIA GPU detected. CUDA $($Matches[1]).$($Matches[2])"
                    } else {
                        Write-Info "NVIDIA GPU detected (could not parse CUDA version, assuming CUDA 12)."
                        $cudaMajor = 12
                    }
                }
            } catch {
                Write-Info "nvidia-smi not found — CPU-only PyTorch will be installed."
            }
        }

        if ($hasCuda -and $cudaMajor -ge 12) {
            Write-Info "Installing PyTorch 2.5.1 + CUDA 12.1..."
            & $VENV_PY -m pip install `
                "torch==2.5.1+cu121" `
                "torchvision==0.20.1+cu121" `
                --index-url https://download.pytorch.org/whl/cu121
        } elseif ($hasCuda -and $cudaMajor -ge 11) {
            Write-Info "Installing PyTorch 2.5.1 + CUDA 11.8..."
            & $VENV_PY -m pip install `
                "torch==2.5.1+cu118" `
                "torchvision==0.20.1+cu118" `
                --index-url https://download.pytorch.org/whl/cu118
        } else {
            Write-Info "Installing PyTorch (CPU only)..."
            & $VENV_PY -m pip install torch torchvision `
                --index-url https://download.pytorch.org/whl/cpu
        }

        $torchCheck = & $VENV_PY -c "import torch; print(torch.__version__)" 2>&1
        if ($torchCheck -match "error|Error|No module") {
            Exit-WithError "PyTorch installation failed: $torchCheck"
        }
        Write-OK "PyTorch installed: $torchCheck"
    }
}

# ==============================================================================
# STEP 4 — Python packages
# ==============================================================================
Write-Step 4 $TOTAL_STEPS "Installing Python packages..."

# ── Core backend requirements ─────────────────────────────────────────────────
if (-not (Test-Path $BACKEND_REQ)) {
    Exit-WithError "Requirements file not found: $BACKEND_REQ"
}

# Build list: optionally strip segment-geospatial/SAM if --SkipSam
if ($SkipSam) {
    Write-Warn "--SkipSam: skipping segment-geospatial and triton-windows."
    $filteredReq = Join-Path $TEMP_DIR "req_nosam.txt"
    Get-Content $BACKEND_REQ |
        Where-Object { $_ -notmatch "segment.geospatial|triton.windows|samgeo" } |
        Set-Content $filteredReq
    $reqFile = $filteredReq
} else {
    $reqFile = $BACKEND_REQ
}

Write-Info "Installing from: $reqFile"
Write-Info "(This may take several minutes — packages are large)"
& $VENV_PY -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Exit-WithError "pip install failed (exit code $LASTEXITCODE). Check output above."
}
Write-OK "Core packages installed."

# ── GPU extras (cupy + CUDA runtime DLLs) ────────────────────────────────────
if (-not $CPU -and (Test-Path $GPU_REQ)) {
    try {
        $gpuName = & "nvidia-smi" "--query-gpu=name" "--format=csv,noheader" 2>&1
        if ($gpuName -notmatch "error|not found|is not recognized") {
            Write-Info "NVIDIA GPU detected. Installing GPU extras (cupy + CUDA DLLs)..."
            & $VENV_PY -m pip install -r $GPU_REQ
            if ($LASTEXITCODE -eq 0) {
                Write-OK "GPU extras installed."
            } else {
                Write-Warn "GPU extras installation had errors (non-fatal, falling back to CPU KMeans)."
            }
        }
    } catch {
        Write-Info "No NVIDIA GPU — skipping GPU extras."
    }
}

# ── Verify key imports ───────────────────────────────────────────────────────
Write-Info "Verifying key imports..."
$imports = @("fastapi", "uvicorn", "numpy", "sklearn", "rasterio", "geopandas", "faiss")
$failedImports = @()
foreach ($mod in $imports) {
    $check = & $VENV_PY -c "import $mod" 2>&1
    if ($check -match "No module|Error") {
        $failedImports += $mod
        Write-Warn "  MISSING: $mod"
    }
}
if ($failedImports.Count -gt 0) {
    Write-Warn "Some imports failed: $($failedImports -join ', ')"
    Write-Warn "The app may still work — check the output above for details."
} else {
    Write-OK "All key packages imported successfully."
}

# ==============================================================================
# STEP 5 — Node.js + Frontend build
# ==============================================================================
Write-Step 5 $TOTAL_STEPS "Setting up frontend..."

if ($SkipFrontend) {
    Write-Warn "Skipping frontend build (--SkipFrontend flag set)."
} elseif (Test-Path $DIST_INDEX) {
    Write-OK "Frontend already built at web_app\dist\ — skipping."
} else {
    Write-Info "Frontend not built yet. Need Node.js..."

    # Check for Node.js
    $nodeCmd = $null
    try {
        $nodeVer = & node --version 2>&1
        if ($nodeVer -match "v\d+") { $nodeCmd = "node"; Write-Info "Node.js found: $nodeVer" }
    } catch {}

    if (-not $nodeCmd) {
        Write-Warn "Node.js not found. Installing via winget..."
        $wingetOk = $false
        try {
            $wgCheck = & winget --version 2>&1
            if ($wgCheck -match "\d+\.\d+") {
                & winget install OpenJS.NodeJS.LTS `
                    --silent `
                    --accept-package-agreements `
                    --accept-source-agreements
                $wingetOk = ($LASTEXITCODE -eq 0)
            }
        } catch {}

        if (-not $wingetOk) {
            Write-Info "winget unavailable. Downloading Node.js LTS installer..."
            $nodeUrl = "https://nodejs.org/dist/v20.19.0/node-v20.19.0-x64.msi"
            $nodeMsi = Join-Path $TEMP_DIR "node-v20.19.0-x64.msi"
            Write-Info "Downloading: $nodeUrl"
            try {
                $wc = New-Object System.Net.WebClient
                $wc.DownloadFile($nodeUrl, $nodeMsi)
            } catch {
                Exit-WithError "Failed to download Node.js: $_"
            }
            Write-Info "Installing Node.js (silent)..."
            $proc = Start-Process msiexec -ArgumentList "/i `"$nodeMsi`" /quiet /norestart" -Wait -PassThru
            if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
                Exit-WithError "Node.js installer failed with exit code $($proc.ExitCode)."
            }
        }

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" `
                  + [System.Environment]::GetEnvironmentVariable("Path","User")

        # Verify
        try {
            $nodeVer = & node --version 2>&1
            if ($nodeVer -match "v\d+") {
                $nodeCmd = "node"
                Write-OK "Node.js installed: $nodeVer"
            }
        } catch {}

        if (-not $nodeCmd) {
            Exit-WithError "Node.js still not found after install. Restart and re-run, or build the frontend manually:`n  cd web_app && npm install && npm run build"
        }
    }

    # Install npm packages + build
    Write-Info "Running npm install in web_app\ ..."
    Push-Location $WEBAPP_DIR
    try {
        & npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed." }

        Write-Info "Running npm run build..."
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed." }
    } catch {
        Pop-Location
        Exit-WithError "$_"
    }
    Pop-Location

    if (Test-Path $DIST_INDEX) {
        Write-OK "Frontend built successfully at web_app\dist\"
    } else {
        Exit-WithError "Build appeared to succeed but web_app\dist\index.html is missing."
    }
}

# ==============================================================================
# STEP 6 — Final summary
# ==============================================================================
Write-Step 6 $TOTAL_STEPS "Verifying installation..."

$venvPyVer  = & $VENV_PY --version 2>&1
$torchVer   = & $VENV_PY -c "import torch; print(torch.__version__)" 2>&1
$fastapiVer = & $VENV_PY -c "import fastapi; print(fastapi.__version__)" 2>&1
$distExists = Test-Path $DIST_INDEX

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  INSTALLATION COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Python (venv) : $venvPyVer"
Write-Host "  PyTorch       : $torchVer"
Write-Host "  FastAPI       : $fastapiVer"
Write-Host "  Frontend dist : $(if ($distExists) { 'OK — web_app\dist\' } else { 'MISSING (run npm run build)' })"
Write-Host ""
Write-Host "  To launch the app:" -ForegroundColor Cyan
Write-Host "    Double-click  start.bat"
Write-Host "    or run:       .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
Write-Host ""

if (-not (Test-Path (Join-Path $ROOT "models\hf_cache\hub"))) {
    Write-Host "  NOTE: AI model weights not found at models\hf_cache\hub\" -ForegroundColor Yellow
    Write-Host "  Road / building / vegetation extraction features require the HuggingFace" -ForegroundColor Yellow
    Write-Host "  model cache from the development machine (~5 GB)." -ForegroundColor Yellow
    Write-Host "  See STANDALONE_DEPLOYMENT.md for copy instructions." -ForegroundColor Yellow
    Write-Host ""
}

exit 0
