#Requires -Version 5.1
# ============================================================
#  Setup.ps1  —  Classification Web App  Offline Installer
#
#  Fully offline Windows Forms wizard.
#  All packages are bundled — no internet needed on target.
#
#  Folder structure expected next to this script:
#    prerequisites\
#      python-3.11.9-embed-amd64.zip
#      get-pip.py
#    offline_packages\          <- core packages + (optional) CPU torch
#    offline_packages_torch\    <- CUDA torch override  (optional)
#    offline_packages_gpu\      <- cupy / nvidia-cuda-* (optional)
#    app\                       <- pre-built application files
# ============================================================

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

# ── Paths relative to the installer folder ────────────────────────
$ROOT        = $PSScriptRoot
$PREREQ      = Join-Path $ROOT "prerequisites"
$PKG         = Join-Path $ROOT "offline_packages"
$PKG_TORCH   = Join-Path $ROOT "offline_packages_torch"
$PKG_GPU     = Join-Path $ROOT "offline_packages_gpu"
$APP_SRC     = Join-Path $ROOT "app"

$HAS_TORCH   = (Test-Path (Join-Path $PKG_TORCH "*.whl") -PathType Leaf) -or
               (Test-Path (Join-Path $PKG "torch-*.whl")         -PathType Leaf)
$HAS_GPU_PKG = Test-Path (Join-Path $PKG_GPU "cupy_cuda12x-*.whl") -PathType Leaf

# ── UI constants ──────────────────────────────────────────────────
$C_BLUE  = [System.Drawing.Color]::FromArgb(0, 102, 204)
$C_LBLUE = [System.Drawing.Color]::FromArgb(235, 244, 255)
$C_DGRAY = [System.Drawing.Color]::FromArgb(50,  50,  50)
$C_LGRAY = [System.Drawing.Color]::FromArgb(243, 243, 243)
$C_SEP   = [System.Drawing.Color]::FromArgb(210, 210, 210)
$C_GREEN = [System.Drawing.Color]::FromArgb(0,  148, 0)
$C_RED   = [System.Drawing.Color]::DarkRed
$C_WHITE = [System.Drawing.Color]::White
$C_CONS_BG = [System.Drawing.Color]::FromArgb(12, 12, 12)
$C_CONS_FG = [System.Drawing.Color]::FromArgb(178, 220, 178)

$F_TITLE = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
$F_HEAD  = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
$F_BODY  = New-Object System.Drawing.Font("Segoe UI", 9)
$F_SMALL = New-Object System.Drawing.Font("Segoe UI", 8)
$F_MONO  = New-Object System.Drawing.Font("Consolas",  8)

function New-Lbl($text,$x,$y,$w,$h,$font=$F_BODY,$color=$C_DGRAY) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text=$text; $l.Location=[System.Drawing.Point]::new($x,$y)
    $l.Size=[System.Drawing.Size]::new($w,$h); $l.Font=$font
    $l.ForeColor=$color; $l.BackColor=[System.Drawing.Color]::Transparent
    return $l
}

# ══════════════════════════════════════════════════════════════
#  MAIN FORM
# ══════════════════════════════════════════════════════════════
$form = New-Object System.Windows.Forms.Form
$form.Text            = "Classification Web App — Setup"
$form.ClientSize      = [System.Drawing.Size]::new(640, 500)
$form.StartPosition   = "CenterScreen"
$form.FormBorderStyle = "FixedSingle"
$form.MaximizeBox     = $false
$form.BackColor       = $C_WHITE

# ── Header (blue banner) ──────────────────────────────────────
$hdr = New-Object System.Windows.Forms.Panel
$hdr.Dock = "Top"; $hdr.Height = 80; $hdr.BackColor = $C_BLUE

$hdrTitle = New-Lbl "Classification Web App" 20 12 520 30 $F_TITLE $C_WHITE
$hdrSub   = New-Lbl "Setup Wizard — Step 1 of 4" 22 46 520 20 $F_SMALL `
                ([System.Drawing.Color]::FromArgb(200, 225, 255))
$hdr.Controls.AddRange(@($hdrTitle, $hdrSub))

# ── Footer (button bar) ───────────────────────────────────────
$ftr = New-Object System.Windows.Forms.Panel
$ftr.Dock = "Bottom"; $ftr.Height = 56; $ftr.BackColor = $C_LGRAY

$ftrLine = New-Object System.Windows.Forms.Label
$ftrLine.Dock="Top"; $ftrLine.Height=1; $ftrLine.BackColor=$C_SEP

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text="Cancel"; $btnCancel.Size=[System.Drawing.Size]::new(90,30)
$btnCancel.Location=[System.Drawing.Point]::new(538,13); $btnCancel.FlatStyle="System"

$btnBack = New-Object System.Windows.Forms.Button
$btnBack.Text="< Back"; $btnBack.Size=[System.Drawing.Size]::new(90,30)
$btnBack.Location=[System.Drawing.Point]::new(438,13); $btnBack.FlatStyle="System"
$btnBack.Enabled=$false

$btnNext = New-Object System.Windows.Forms.Button
$btnNext.Text="Next >"; $btnNext.Size=[System.Drawing.Size]::new(90,30)
$btnNext.Location=[System.Drawing.Point]::new(338,13); $btnNext.FlatStyle="System"
$form.AcceptButton=$btnNext

$ftr.Controls.AddRange(@($ftrLine,$btnCancel,$btnBack,$btnNext))

# ── Content pane ─────────────────────────────────────────────
$cnt = New-Object System.Windows.Forms.Panel
$cnt.Location=[System.Drawing.Point]::new(0,80)
$cnt.Size=[System.Drawing.Size]::new(640,364)    # 500 - 80 hdr - 56 ftr
$cnt.BackColor=$C_WHITE

$form.Controls.AddRange(@($hdr,$ftr,$cnt))

# ══════════════════════════════════════════════════════════════
#  PAGE 0 — WELCOME
# ══════════════════════════════════════════════════════════════
$p0 = New-Object System.Windows.Forms.Panel
$p0.Dock="Fill"; $p0.BackColor=$C_WHITE

$p0.Controls.Add((New-Lbl "Welcome" 30 22 560 28 $F_HEAD $C_BLUE))
$p0_body = New-Object System.Windows.Forms.Label
$p0_body.Location=[System.Drawing.Point]::new(30,58)
$p0_body.Size=[System.Drawing.Size]::new(568,140)
$p0_body.Font=$F_BODY; $p0_body.ForeColor=$C_DGRAY
$p0_body.Text = @"
This wizard installs Classification Web App on this computer.

Everything is bundled — no internet connection is needed.

  Python 3.11.9    (portable, no system installation required)
  FastAPI backend  (with all required libraries)
  Web frontend     (pre-built)
"@

if ($HAS_TORCH) {
    $p0_body.Text += "  PyTorch          (for AI road / building / vegetation extraction)`n"
}
if ($HAS_GPU_PKG) {
    $p0_body.Text += "  CuPy + NVIDIA    (GPU-accelerated KMeans)`n"
}

$p0.Controls.Add($p0_body)
$p0.Controls.Add((New-Lbl "Estimated disk space required:  8 – 12 GB" 30 205 500 20 $F_SMALL $C_DGRAY))

$p0box = New-Object System.Windows.Forms.Panel
$p0box.Location=[System.Drawing.Point]::new(30,240); $p0box.Size=[System.Drawing.Size]::new(568,54)
$p0box.BackColor=$C_LBLUE; $p0box.BorderStyle="FixedSingle"
$p0box.Controls.Add((New-Lbl "AI model weights (~5 GB) are bundled in this installer package." 12 10 540 18 $F_SMALL $C_DGRAY))
$p0box.Controls.Add((New-Lbl "They will be copied automatically — no manual steps required." 12 28 540 18 $F_SMALL $C_DGRAY))
$p0.Controls.Add($p0box)

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — INSTALL LOCATION
# ══════════════════════════════════════════════════════════════
$p1 = New-Object System.Windows.Forms.Panel
$p1.Dock="Fill"; $p1.BackColor=$C_WHITE; $p1.Visible=$false

$p1.Controls.Add((New-Lbl "Install Location" 30 22 560 28 $F_HEAD $C_BLUE))
$p1.Controls.Add((New-Lbl "Where should Classification Web App be installed?" 30 58 560 20))
$p1.Controls.Add((New-Lbl "Tip: avoid paths with spaces. Recommended: C:\ClassificationApp" 30 78 560 18 $F_SMALL $C_DGRAY))

$p1_path = New-Object System.Windows.Forms.TextBox
$p1_path.Location=[System.Drawing.Point]::new(30,118); $p1_path.Size=[System.Drawing.Size]::new(476,24)
$p1_path.Font=$F_BODY; $p1_path.Text="C:\ClassificationApp"

$p1_browse = New-Object System.Windows.Forms.Button
$p1_browse.Text="Browse..."; $p1_browse.Size=[System.Drawing.Size]::new(84,26)
$p1_browse.Location=[System.Drawing.Point]::new(516,119); $p1_browse.FlatStyle="System"
$p1_browse.Add_Click({
    $d=New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description="Select installation folder"; $d.SelectedPath=$p1_path.Text
    $d.ShowNewFolderButton=$true
    if ($d.ShowDialog() -eq "OK") { $p1_path.Text=$d.SelectedPath }
})

$p1.Controls.AddRange(@($p1_path,$p1_browse))
$p1.Controls.Add((New-Lbl "Estimated disk space required:  8 – 12 GB" 30 158 500 20 $F_SMALL $C_DGRAY))

$p1_desk = New-Object System.Windows.Forms.CheckBox
$p1_desk.Text="Create Desktop shortcut"
$p1_desk.Location=[System.Drawing.Point]::new(30,210); $p1_desk.Size=[System.Drawing.Size]::new(300,24)
$p1_desk.Font=$F_BODY; $p1_desk.Checked=$true
$p1.Controls.Add($p1_desk)

$p1_menu = New-Object System.Windows.Forms.CheckBox
$p1_menu.Text="Create Start Menu shortcut"
$p1_menu.Location=[System.Drawing.Point]::new(30,238); $p1_menu.Size=[System.Drawing.Size]::new(300,24)
$p1_menu.Font=$F_BODY; $p1_menu.Checked=$true
$p1.Controls.Add($p1_menu)

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — OPTIONS
# ══════════════════════════════════════════════════════════════
$p2 = New-Object System.Windows.Forms.Panel
$p2.Dock="Fill"; $p2.BackColor=$C_WHITE; $p2.Visible=$false

$p2.Controls.Add((New-Lbl "Installation Options" 30 22 560 28 $F_HEAD $C_BLUE))

# Feature pack group
$p2_feat = New-Object System.Windows.Forms.GroupBox
$p2_feat.Text="Feature Pack"; $p2_feat.Font=$F_BODY
$p2_feat.Location=[System.Drawing.Point]::new(30,58); $p2_feat.Size=[System.Drawing.Size]::new(566,90)

$p2_full = New-Object System.Windows.Forms.RadioButton
$p2_full.Text="Full install  —  classification + AI road / building / vegetation extraction"
$p2_full.Location=[System.Drawing.Point]::new(12,22); $p2_full.Size=[System.Drawing.Size]::new(540,22)
$p2_full.Font=$F_BODY; $p2_full.Checked=$true

$p2_core = New-Object System.Windows.Forms.RadioButton
$p2_core.Text="Core only  —  classification only  (~3 GB smaller, faster install)"
$p2_core.Location=[System.Drawing.Point]::new(12,50); $p2_core.Size=[System.Drawing.Size]::new(540,22)
$p2_core.Font=$F_BODY

$p2_feat.Controls.AddRange(@($p2_full,$p2_core))
$p2.Controls.Add($p2_feat)

# Torch variant group (show only if both CPU and CUDA wheels present)
$p2_torch = New-Object System.Windows.Forms.GroupBox
$p2_torch.Text="PyTorch Variant"; $p2_torch.Font=$F_BODY
$p2_torch.Location=[System.Drawing.Point]::new(30,162); $p2_torch.Size=[System.Drawing.Size]::new(566,90)
$p2_torch.Enabled = $HAS_TORCH

$p2_t_auto = New-Object System.Windows.Forms.RadioButton
$p2_t_auto.Text="Auto  (use whatever was bundled — recommended)"
$p2_t_auto.Location=[System.Drawing.Point]::new(12,22); $p2_t_auto.Size=[System.Drawing.Size]::new(540,22)
$p2_t_auto.Font=$F_BODY; $p2_t_auto.Checked=$true

$p2_t_cuda = New-Object System.Windows.Forms.RadioButton
$p2_t_cuda.Text="Force CUDA  (override with CUDA wheels from offline_packages_torch)"
$p2_t_cuda.Location=[System.Drawing.Point]::new(12,50); $p2_t_cuda.Size=[System.Drawing.Size]::new(540,22)
$p2_t_cuda.Font=$F_BODY
$p2_t_cuda.Enabled=(Test-Path (Join-Path $PKG_TORCH "*.whl") -PathType Leaf)

if (-not $HAS_TORCH) {
    $p2_torch.Text = "PyTorch Variant  (not bundled — AI features unavailable)"
}

$p2_torch.Controls.AddRange(@($p2_t_auto,$p2_t_cuda))
$p2.Controls.Add($p2_torch)

# GPU acceleration (cupy) — only show if packages exist
$p2_gpu = New-Object System.Windows.Forms.GroupBox
$p2_gpu.Text="GPU KMeans Acceleration  (CuPy)"; $p2_gpu.Font=$F_BODY
$p2_gpu.Location=[System.Drawing.Point]::new(30,266); $p2_gpu.Size=[System.Drawing.Size]::new(566,68)
$p2_gpu.Enabled=$HAS_GPU_PKG

$p2_gpu_yes = New-Object System.Windows.Forms.CheckBox
$p2_gpu_yes.Text = if ($HAS_GPU_PKG) { "Install CuPy + NVIDIA packages for GPU-accelerated KMeans" } `
                   else { "Not bundled  (run prepare_offline.bat with GPU option to include)" }
$p2_gpu_yes.Location=[System.Drawing.Point]::new(12,26); $p2_gpu_yes.Size=[System.Drawing.Size]::new(540,22)
$p2_gpu_yes.Font=$F_BODY; $p2_gpu_yes.Checked=$HAS_GPU_PKG

$p2_gpu.Controls.Add($p2_gpu_yes)
$p2.Controls.Add($p2_gpu)

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — INSTALLING
# ══════════════════════════════════════════════════════════════
$p3 = New-Object System.Windows.Forms.Panel
$p3.Dock="Fill"; $p3.BackColor=$C_WHITE; $p3.Visible=$false

$p3.Controls.Add((New-Lbl "Installing..." 30 22 560 28 $F_HEAD $C_BLUE))

$p3_status = New-Lbl "Preparing..." 30 58 570 18 $F_SMALL $C_DGRAY
$p3.Controls.Add($p3_status)

$p3_bar = New-Object System.Windows.Forms.ProgressBar
$p3_bar.Location=[System.Drawing.Point]::new(30,80); $p3_bar.Size=[System.Drawing.Size]::new(566,20)
$p3_bar.Minimum=0; $p3_bar.Maximum=100; $p3_bar.Style="Continuous"
$p3.Controls.Add($p3_bar)

$p3_log = New-Object System.Windows.Forms.RichTextBox
$p3_log.Location=[System.Drawing.Point]::new(30,110); $p3_log.Size=[System.Drawing.Size]::new(566,220)
$p3_log.Font=$F_MONO; $p3_log.BackColor=$C_CONS_BG; $p3_log.ForeColor=$C_CONS_FG
$p3_log.ReadOnly=$true; $p3_log.ScrollBars="Vertical"; $p3_log.WordWrap=$false
$p3.Controls.Add($p3_log)

$p3_warn = New-Lbl "Do not close this window. Installation may take 10 - 30 minutes." `
               30 340 566 18 $F_SMALL $C_RED
$p3.Controls.Add($p3_warn)

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — DONE
# ══════════════════════════════════════════════════════════════
$p4 = New-Object System.Windows.Forms.Panel
$p4.Dock="Fill"; $p4.BackColor=$C_WHITE; $p4.Visible=$false

$p4_icon = New-Object System.Windows.Forms.Label
$p4_icon.Text=[char]0x2714; $p4_icon.Font=New-Object System.Drawing.Font("Segoe UI",52,[System.Drawing.FontStyle]::Bold)
$p4_icon.ForeColor=$C_GREEN; $p4_icon.Location=[System.Drawing.Point]::new(28,22)
$p4_icon.Size=[System.Drawing.Size]::new(72,72)
$p4.Controls.Add($p4_icon)

$p4.Controls.Add((New-Lbl "Installation complete!" 108 38 420 32 $F_TITLE $C_GREEN))

$p4_info = New-Object System.Windows.Forms.Label
$p4_info.Location=[System.Drawing.Point]::new(30,110); $p4_info.Size=[System.Drawing.Size]::new(568,90)
$p4_info.Font=$F_BODY; $p4_info.ForeColor=$C_DGRAY
$p4.Controls.Add($p4_info)

$p4_launch = New-Object System.Windows.Forms.CheckBox
$p4_launch.Text="Launch Classification Web App now"
$p4_launch.Location=[System.Drawing.Point]::new(30,218); $p4_launch.Size=[System.Drawing.Size]::new(400,24)
$p4_launch.Font=$F_BODY; $p4_launch.Checked=$true
$p4.Controls.Add($p4_launch)

$p4_model = New-Object System.Windows.Forms.Label
$p4_model.Font=$F_SMALL; $p4_model.ForeColor=[System.Drawing.Color]::DarkGreen
$p4_model.Location=[System.Drawing.Point]::new(30,260); $p4_model.Size=[System.Drawing.Size]::new(568,36)
$p4_model.Text = "To enable AI road/building/vegetation extraction, copy the SAM model weights:`n" +
                  "  From (dev machine):  C:\Users\<name>\.cache\huggingface\hub\`n" +
                  "  Into (this machine):  <install folder>\models\hf_cache\hub\"
$p4_model.ForeColor = [System.Drawing.Color]::FromArgb(180, 100, 0)
$p4.Controls.Add($p4_model)

$cnt.Controls.AddRange(@($p0,$p1,$p2,$p3,$p4))

# ══════════════════════════════════════════════════════════════
#  PAGE NAVIGATION
# ══════════════════════════════════════════════════════════════
$PAGES     = @($p0,$p1,$p2,$p3,$p4)
$PAGE_SUBS = @(
    "Step 1 of 4 — Welcome",
    "Step 2 of 4 — Install Location",
    "Step 3 of 4 — Options",
    "Step 4 of 4 — Installing",
    "Installation Complete"
)
$script:page = 0

function Set-Page([int]$i) {
    $script:page = $i
    foreach ($p in $PAGES) { $p.Visible = $false }
    $PAGES[$i].Visible = $true
    $hdrSub.Text = $PAGE_SUBS[$i]
    $btnBack.Enabled = ($i -gt 0 -and $i -ne 3 -and $i -ne 4)
    switch ($i) {
        3 { $btnNext.Text="Installing..."; $btnNext.Enabled=$false }
        4 { $btnNext.Text="Finish";        $btnNext.Enabled=$true  }
        default { $btnNext.Text="Next >"; $btnNext.Enabled=$true }
    }
}
Set-Page 0

# ══════════════════════════════════════════════════════════════
#  INSTALL HELPERS
# ══════════════════════════════════════════════════════════════
function Log([string]$msg, [string]$color="White") {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg
    $p3_log.Invoke([Action]{
        $p3_log.SelectionColor = [System.Drawing.Color]::FromName($color)
        $p3_log.AppendText("$line`n")
        $p3_log.ScrollToCaret()
    })
}

function SetStatus([string]$msg) {
    $p3_status.Invoke([Action]{ $p3_status.Text = $msg })
}

function SetPct([int]$v) {
    $p3_bar.Invoke([Action]{ $p3_bar.Value = [Math]::Min($v, 100) })
}

# Run an external process, stream its stdout/stderr into the log box.
# Returns the exit code.
function Invoke-Step([string]$exe, [string]$args, [string]$label) {
    Log ">> $label" "Cyan"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName=$exe; $psi.Arguments=$args
    $psi.UseShellExecute=$false
    $psi.RedirectStandardOutput=$true
    $psi.RedirectStandardError=$true
    $psi.CreateNoWindow=$true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo=$psi
    $proc.add_OutputDataReceived({ param($s,$e)
        if ($e.Data) { Log "   $($e.Data)" "Gray" }
    })
    $proc.add_ErrorDataReceived({ param($s,$e)
        if ($e.Data) { Log "   $($e.Data)" "Yellow" }
    })
    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.WaitForExit()
    return $proc.ExitCode
}

# ══════════════════════════════════════════════════════════════
#  MAIN INSTALL ROUTINE  (runs on a background thread)
# ══════════════════════════════════════════════════════════════
function Start-Install([string]$dir, [bool]$skipAI, [bool]$forceCuda, [bool]$installGpu) {

    $pyDir  = Join-Path $dir "python"
    $pyExe  = Join-Path $pyDir "python.exe"

    # ── 1. Locate / copy Python 3.11 ─────────────────────────
    SetStatus "Step 1 / 9 — Locating Python 3.11..."; SetPct 2
    Log ""; Log "=== [1/9] Python 3.11 ===" "White"

    # Try to find an existing Python 3.11 on the target machine first
    $existingPy = $null
    $candidates = @(
        "python",
        "py -3.11"
    )
    # Also check common install paths
    $commonPaths = @(
        "$env:LocalAppData\Python\pythoncore-3.11-64\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($p in $commonPaths) {
        if (Test-Path $p) { $candidates += $p }
    }

    foreach ($c in $candidates) {
        $ver = & cmd /c "$c --version 2>&1" 2>$null
        if ($ver -match "3\.11") { $existingPy = $c; break }
    }

    if ($existingPy) {
        # Python 3.11 already installed — create a clean venv in the install folder
        Log "Found existing Python 3.11: $existingPy" "Green"
        $venvDir = Join-Path $dir ".venv"
        Log "Creating venv at: $venvDir" "Cyan"
        $proc = Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/c `"$existingPy`" -m venv `"$venvDir`"" `
            -Wait -PassThru -NoNewWindow
        $pyExe = Join-Path $venvDir "Scripts\python.exe"
        $pyDir = $venvDir
        if (-not (Test-Path $pyExe)) {
            Log "ERROR: venv creation failed (exit $($proc.ExitCode))." "Red"; return $false
        }
        Log "venv ready." "Green"
    } else {
        # Not found — copy from bundled prerequisites
        Log "Python 3.11 not found on this machine. Copying from installer bundle..." "Yellow"
        $pythonSrc = Join-Path $PREREQ "python311"
        $zipPath   = Join-Path $PREREQ "python-3.11.9-embed-amd64.zip"
        $pyDir     = Join-Path $dir "python"

        if (Test-Path (Join-Path $pythonSrc "python.exe")) {
            if (Test-Path $pyDir) { Remove-Item $pyDir -Recurse -Force }
            Copy-Item $pythonSrc $pyDir -Recurse -Force
            Log "Python copied to: $pyDir" "Green"
        } elseif (Test-Path $zipPath) {
            if (Test-Path $pyDir) { Remove-Item $pyDir -Recurse -Force }
            New-Item -ItemType Directory -Path $pyDir | Out-Null
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $pyDir)
            Log "Python extracted from zip to: $pyDir" "Green"
        } else {
            Log "ERROR: Python 3.11 not found on machine and no bundle in prerequisites\." "Red"
            return $false
        }
        $pyExe = Join-Path $pyDir "python.exe"
    }

    # ── 2. Enable site-packages (only needed for embeddable zip) ─
    SetStatus "Step 2 / 9 — Configuring Python..."; SetPct 7
    Log ""; Log "=== [2/9] Configuring Python ===" "White"

    $pth = Get-ChildItem $pyDir -Filter "*._pth" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pth) {
        $txt = [System.IO.File]::ReadAllText($pth.FullName)
        if ($txt -match '#import site') {
            $txt = $txt -replace '(?m)^#import site', 'import site'
            [System.IO.File]::WriteAllText($pth.FullName, $txt)
            Log "Patched $($pth.Name) to enable site-packages" "Green"
        } else {
            Log "site-packages already enabled" "Gray"
        }
    }

    # ── 3. Verify / upgrade pip ───────────────────────────────
    SetStatus "Step 3 / 9 — Checking pip..."; SetPct 12
    Log ""; Log "=== [3/9] Verifying pip ===" "White"

    $pipCheck = & $pyExe -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log "ERROR: pip not available in Python environment." "Red"; return $false
    }
    Log "pip OK: $pipCheck" "Gray"

    # Upgrade pip/setuptools/wheel only if wheels exist in offline cache
    $pipWhl = Get-ChildItem $PKG -Filter "pip-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pipWhl) {
        Invoke-Step $pyExe "-m pip install --no-index --find-links `"$PKG`" --upgrade pip setuptools wheel" "upgrade pip" | Out-Null
        Log "pip upgraded from offline cache." "Gray"
    } else {
        Log "pip wheels not in cache — using installed pip as-is." "Gray"
    }
    Log "pip ready." "Green"

    # ── 4. Build a filtered core requirements file ────────────
    SetStatus "Step 4 / 9 — Installing core packages..."; SetPct 18
    Log ""; Log "=== [4/9] Core packages ===" "White"

    # Filter out comment-only lines, blank lines, SAM packages.
    # torch is also excluded here — we install it specifically in step 5.
    $reqSrc = Join-Path $APP_SRC "requirements.txt"
    $tmpReq = Join-Path $env:TEMP "cwa_core_req.txt"
    if (Test-Path $reqSrc) {
        Get-Content $reqSrc |
            Where-Object { $_ -notmatch '^\s*#' } |
            Where-Object { $_ -notmatch 'segment.geospatial' } |
            Where-Object { $_ -notmatch 'triton.windows' } |
            Where-Object { $_ -notmatch 'opencv.python' } |
            Where-Object { $_.Trim() -ne '' } |
            Set-Content $tmpReq
        Log "Core requirements written to temp file." "Gray"
        $rc = Invoke-Step $pyExe `
            "-m pip install --no-index --find-links `"$PKG`" -r `"$tmpReq`"" `
            "pip install core requirements"
        if ($rc -ne 0) { Log "WARNING: some core packages failed (exit $rc)" "Yellow" }
        Remove-Item $tmpReq -Force -ErrorAction SilentlyContinue
    } else {
        Log "requirements.txt not found — skipping core install" "Yellow"
    }
    Log "Core packages done." "Green"; SetPct 45

    # ── 5. PyTorch ────────────────────────────────────────────
    if (-not $skipAI -and $HAS_TORCH) {
        SetStatus "Step 5 / 9 — Installing PyTorch..."; SetPct 48
        Log ""; Log "=== [5/9] PyTorch ===" "White"

        # Decide which source to use: CUDA override or the bundled wheels
        $torchSrc = if ($forceCuda -and (Test-Path (Join-Path $PKG_TORCH "*.whl") -PathType Leaf)) {
            "$PKG_TORCH"
        } else {
            "$PKG"
        }
        $rc = Invoke-Step $pyExe `
            "-m pip install --no-index --find-links `"$torchSrc`" torch torchvision" `
            "pip install torch torchvision"
        if ($rc -ne 0) { Log "WARNING: PyTorch install returned $rc" "Yellow" }
        Log "PyTorch done." "Green"; SetPct 65
    } elseif ($skipAI) {
        Log ""; Log "=== [5/9] PyTorch — SKIPPED (Core only) ===" "Yellow"
        SetPct 65
    } else {
        Log ""; Log "=== [5/9] PyTorch — SKIPPED (not bundled) ===" "Yellow"
        SetPct 65
    }

    # ── 6. SAM / AI packages ──────────────────────────────────
    if (-not $skipAI -and $HAS_TORCH) {
        SetStatus "Step 6 / 9 — Installing AI / SAM packages..."; SetPct 67
        Log ""; Log "=== [6/9] SAM / AI packages ===" "White"

        $samPkgs = "segment-geospatial[samgeo3] triton-windows opencv-python-headless"
        $rc = Invoke-Step $pyExe `
            "-m pip install --no-index --find-links `"$PKG`" $samPkgs" `
            "pip install SAM packages"
        if ($rc -ne 0) { Log "WARNING: SAM packages returned $rc" "Yellow" }
        Log "SAM packages done." "Green"; SetPct 80
    } else {
        Log ""; Log "=== [6/9] SAM — SKIPPED ===" "Yellow"
        SetPct 80
    }

    # ── 7. GPU KMeans packages ────────────────────────────────
    if ($installGpu -and $HAS_GPU_PKG) {
        SetStatus "Step 7 / 9 — Installing GPU packages (CuPy)..."; SetPct 82
        Log ""; Log "=== [7/9] GPU KMeans (CuPy) ===" "White"

        $gpuPkgs = "cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12"
        $rc = Invoke-Step $pyExe `
            "-m pip install --no-index --find-links `"$PKG_GPU`" $gpuPkgs" `
            "pip install cupy/nvidia packages"
        if ($rc -ne 0) { Log "WARNING: GPU packages returned $rc" "Yellow" }
        Log "GPU packages done." "Green"; SetPct 86
    } else {
        Log ""; Log "=== [7/9] GPU packages — SKIPPED ===" "Yellow"
        SetPct 86
    }

    # ── 8. Copy application files ─────────────────────────────
    SetStatus "Step 8 / 9 — Copying application files..."; SetPct 88
    Log ""; Log "=== [8/9] Copying app files ===" "White"

    function Copy-Dir([string]$src, [string]$dst) {
        if (Test-Path $src) {
            if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst | Out-Null }
            Copy-Item "$src\*" $dst -Recurse -Force
            Log "Copied: $src" "Gray"
        } else { Log "WARNING: source not found: $src" "Yellow" }
    }

    Copy-Dir (Join-Path $APP_SRC "backend")    (Join-Path $dir "backend")
    Copy-Dir (Join-Path $APP_SRC "web_app")    (Join-Path $dir "web_app")
    Copy-Dir (Join-Path $APP_SRC "shared")     (Join-Path $dir "shared")

    foreach ($f in @("launcher.py","requirements.txt","requirements-gpu.txt","version.txt")) {
        $s = Join-Path $APP_SRC $f
        if (Test-Path $s) { Copy-Item $s $dir -Force }
    }

    # Copy HuggingFace model weights if bundled
    $hfSrc = Join-Path $APP_SRC "models\hf_cache\hub"
    $hfDst = Join-Path $dir "models\hf_cache\hub"
    if (Test-Path $hfSrc) {
        SetStatus "Copying AI model weights (~5 GB)..."
        Log "" "White"
        Log "=== Copying AI model weights ===" "White"
        if (-not (Test-Path $hfDst)) { New-Item -ItemType Directory -Path $hfDst | Out-Null }
        Copy-Item "$hfSrc\*" $hfDst -Recurse -Force
        Log "Model weights copied." "Green"
    } else {
        New-Item -ItemType Directory -Path $hfDst -Force | Out-Null
        Log "Model weights not bundled — AI extraction will not work." "Yellow"
    }

    # Copy STANDALONE_DEPLOYMENT.md if it exists next to Setup.ps1
    $depDoc = Join-Path $ROOT "STANDALONE_DEPLOYMENT.md"
    if (Test-Path $depDoc) { Copy-Item $depDoc $dir -Force }

    Log "App files copied." "Green"; SetPct 93

    # ── 9. Write start.bat + shortcuts ───────────────────────
    SetStatus "Step 9 / 9 — Finalizing..."; SetPct 95
    Log ""; Log "=== [9/9] Writing launch script and shortcuts ===" "White"

    # Write a start.bat that uses the embedded Python in <install dir>\python\
    $startBatContent = "@echo off`r`n" +
        "cd /d `"%~dp0`"`r`n" +
        "echo Starting Classification Web App...`r`n" +
        "set HF_HUB_OFFLINE=1`r`n" +
        "set TRANSFORMERS_OFFLINE=1`r`n" +
        "set HF_HOME=%~dp0models\hf_cache`r`n" +
        "set PYTHONPATH=%~dp0`r`n" +
        "start `"`" http://127.0.0.1:8000`r`n" +
        "python\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1`r`n"
    [System.IO.File]::WriteAllText((Join-Path $dir "start.bat"), $startBatContent,
        [System.Text.Encoding]::ASCII)
    Log "start.bat written." "Green"

    $wsh = New-Object -ComObject WScript.Shell
    $startBatPath = Join-Path $dir "start.bat"

    if ($script:wantDesktop) {
        $lnk = $wsh.CreateShortcut(
            [System.IO.Path]::Combine(
                [System.Environment]::GetFolderPath("Desktop"),
                "Classification Web App.lnk"))
        $lnk.TargetPath=$startBatPath; $lnk.WorkingDirectory=$dir
        $lnk.Description="Launch Classification Web App"; $lnk.Save()
        Log "Desktop shortcut created." "Green"
    }

    if ($script:wantMenu) {
        $smDir = Join-Path ([System.Environment]::GetFolderPath("Programs")) "Classification Web App"
        if (-not (Test-Path $smDir)) { New-Item -ItemType Directory -Path $smDir | Out-Null }
        $lnk = $wsh.CreateShortcut("$smDir\Classification Web App.lnk")
        $lnk.TargetPath=$startBatPath; $lnk.WorkingDirectory=$dir; $lnk.Save()
        Log "Start Menu shortcut created." "Green"
    }

    SetPct 100
    return $true
}

# ══════════════════════════════════════════════════════════════
#  BUTTON LOGIC
# ══════════════════════════════════════════════════════════════
$btnCancel.Add_Click({
    if ($script:page -eq 3) {
        [System.Windows.Forms.MessageBox]::Show(
            "Installation is running. Please wait for it to complete.",
            "Setup", "OK", "Warning") | Out-Null
        return
    }
    $r = [System.Windows.Forms.MessageBox]::Show(
        "Cancel the installation?", "Cancel Setup", "YesNo", "Question")
    if ($r -eq "Yes") { $form.Close() }
})

$btnBack.Add_Click({
    if ($script:page -gt 0 -and $script:page -ne 3) { Set-Page ($script:page - 1) }
})

$btnNext.Add_Click({
    # ── Validate page 1 ───────────────────────────────────────
    if ($script:page -eq 1) {
        $dir = $p1_path.Text.Trim()
        if ([string]::IsNullOrWhiteSpace($dir)) {
            [System.Windows.Forms.MessageBox]::Show(
                "Please choose an installation folder.", "Required", "OK", "Warning") | Out-Null
            return
        }
    }

    # ── Finish ────────────────────────────────────────────────
    if ($script:page -eq 4) {
        if ($p4_launch.Checked) {
            $sb = Join-Path $p1_path.Text.Trim() "start.bat"
            if (Test-Path $sb) { Start-Process "cmd.exe" "/c `"$sb`"" }
        }
        $form.Close(); return
    }

    # ── Kick off installation when leaving page 2 ─────────────
    if ($script:page -eq 2) {
        Set-Page 3

        $installDir  = $p1_path.Text.Trim()
        $script:wantDesktop = $p1_desk.Checked
        $script:wantMenu    = $p1_menu.Checked
        $skipAI      = $p2_core.Checked
        $forceCuda   = $p2_t_cuda.Checked
        $installGpu  = $p2_gpu_yes.Checked -and $HAS_GPU_PKG

        if (-not (Test-Path $installDir)) {
            New-Item -ItemType Directory -Path $installDir -Force | Out-Null
        }

        # Run on background thread so the UI stays responsive
        $thread = [System.Threading.Thread]::new([System.Threading.ThreadStart]{
            $ok = $false
            try {
                $ok = Start-Install -dir $installDir -skipAI $skipAI `
                                    -forceCuda $forceCuda -installGpu $installGpu
            } catch {
                Log "FATAL: $_" "Red"
            }

            if ($ok) {
                Log ""; Log "=== INSTALLATION SUCCESSFUL ===" "LightGreen"
                $p4_info.Invoke([Action]{
                    $p4_info.Text =
                        "Installed to:`r`n`r`n    $installDir`r`n`r`n" +
                        "To launch the app, use the shortcut on your Desktop`r`n" +
                        "or run  start.bat  from the install folder.`r`n" +
                        "The app opens in your browser at  http://127.0.0.1:8000"
                })
                $form.Invoke([Action]{ Set-Page 4 })
            } else {
                Log ""; Log "=== INSTALLATION FAILED — see log above ===" "Red"
                $p3_status.Invoke([Action]{
                    $p3_status.Text      = "Installation failed. See the log above for details."
                    $p3_status.ForeColor = $C_RED
                })
                $btnNext.Invoke([Action]{
                    $btnNext.Text    = "Close"
                    $btnNext.Enabled = $true
                })
                $btnNext.Invoke([Action]{
                    $btnNext.Add_Click({ $form.Close() })
                })
            }
        })
        $thread.IsBackground = $true
        $thread.Start()
        return
    }

    Set-Page ($script:page + 1)
})

# Prevent closing during install
$form.Add_FormClosing({
    param($s,$e)
    if ($script:page -eq 3) { $e.Cancel = $true }
})

[System.Windows.Forms.Application]::Run($form)
