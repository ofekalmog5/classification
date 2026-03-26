; =============================================================================
;  ClassificationApp.iss  —  Inno Setup 6 installer script
;
;  HOW TO BUILD:
;    Run  build_installer.bat  (auto-finds Inno Setup)
;    or:  iscc ClassificationApp.iss
;
;  OUTPUT:
;    dist_installer\ClassificationApp_Setup_1.0.0.exe   (~30–80 MB)
;
;  ON THE TARGET MACHINE:
;    Run the produced .exe — a wizard guides through everything.
;    Internet required during install (downloads Python, pip packages, Node.js).
; =============================================================================

#define AppName      "Classification Web App"
#define AppVersion   "1.0.0"
#define AppPublisher "Classification Tools"
#define AppURL       "http://127.0.0.1:8000"

; ── Basic setup ───────────────────────────────────────────────────────────────
[Setup]
AppId={{6F3A1B2C-4D5E-4F60-A7B8-9C0D1E2F3A4B}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}

; Default install to user-local Program Files (no admin needed)
DefaultDirName={autopf}\ClassificationApp
DefaultGroupName={#AppName}
AllowNoIcons=yes

; Output
OutputDir=dist_installer
OutputBaseFilename=ClassificationApp_Setup_{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes

; Appearance
WizardStyle=modern
WizardResizable=yes
SetupIconFile=

; Privileges — prefer no admin; user can elevate if they want
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Windows 10 1903+ (64-bit)
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.18362

; Misc
DisableProgramGroupPage=yes
ShowLanguageDialog=no
UninstallDisplayIcon={app}\web_app\dist\favicon.ico

; ── Language ──────────────────────────────────────────────────────────────────
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; ── Custom messages ───────────────────────────────────────────────────────────
[CustomMessages]
english.WelcomeLabel2=This wizard will install [name] [ver] on your computer.%n%n\
What will be set up:%n\
  - Python 3.11       (downloaded if not already present)%n\
  - Python virtual environment with all required packages%n\
  - PyTorch           (CUDA or CPU, auto-detected)%n\
  - Node.js           (downloaded if not already present)%n\
  - Frontend build%n%n\
An internet connection and ~15 GB of free disk space are required.%n%n\
It is recommended that you close all other applications before continuing.

; ── Tasks (optional features user can tick) ───────────────────────────────────
[Tasks]
Name: desktopicon;  Description: "{cm:CreateDesktopIcon}";  GroupDescription: "{cm:AdditionalIcons}"
Name: startmenu;    Description: "Create Start Menu shortcuts"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checked

; ── Files to bundle ───────────────────────────────────────────────────────────
[Files]
; Backend (Python API)
Source: "backend\*";          DestDir: "{app}\backend";       Flags: ignoreversion recursesubdirs createallsubdirs

; Frontend source (for optional rebuild on target)
Source: "web_app\src\*";      DestDir: "{app}\web_app\src";   Flags: ignoreversion recursesubdirs createallsubdirs
Source: "web_app\public\*";   DestDir: "{app}\web_app\public"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "web_app\index.html"; DestDir: "{app}\web_app";       Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\package.json";      DestDir: "{app}\web_app"; Flags: ignoreversion
Source: "web_app\package-lock.json"; DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\tsconfig.json";     DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\tsconfig.app.json"; DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\vite.config.ts";    DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\postcss.config.ts"; DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist
Source: "web_app\tailwind.config.ts";DestDir: "{app}\web_app"; Flags: ignoreversion skipifsourcedoesntexist

; Pre-built frontend dist (skip if not present on dev machine — installer will build it)
Source: "web_app\dist\*";     DestDir: "{app}\web_app\dist";  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Shared schemas
Source: "shared\*";           DestDir: "{app}\shared";         Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Launcher & scripts
Source: "launcher.py";        DestDir: "{app}"; Flags: ignoreversion
Source: "start.bat";          DestDir: "{app}"; Flags: ignoreversion
Source: "install.ps1";        DestDir: "{app}"; Flags: ignoreversion

; ── Shortcuts ─────────────────────────────────────────────────────────────────
[Icons]
; Start Menu
Name: "{group}\{#AppName}";         Filename: "{app}\start.bat"; WorkingDir: "{app}"; Tasks: startmenu
Name: "{group}\Uninstall {#AppName}";Filename: "{uninstallexe}";                      Tasks: startmenu

; Desktop
Name: "{autodesktop}\{#AppName}";   Filename: "{app}\start.bat"; WorkingDir: "{app}"; Tasks: desktopicon

; ── Run after install ─────────────────────────────────────────────────────────
[Run]
; Show the PowerShell installer in its own console window so the user can watch
; pip's real-time progress output (installation may take 15–30 minutes).
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" {code:GetPsFlags}"; \
  WorkingDir: "{app}"; \
  StatusMsg: "Installing Python packages — this may take 15-30 minutes..."; \
  Description: "Set up Python environment"; \
  Flags: waituntilterminated

; ── Uninstall cleanup ─────────────────────────────────────────────────────────
[UninstallRun]
; Remove the venv on uninstall (optional — can be large)
Filename: "cmd.exe"; Parameters: "/c rmdir /s /q ""{app}\.venv"""; Flags: runhidden; RunOnceId: "DelVenv"

; ── Pascal code — custom wizard pages ─────────────────────────────────────────
[Code]

// ─── Variables ───────────────────────────────────────────────────────────────
var
  GpuPage:   TInputOptionWizardPage;   // GPU / PyTorch choice
  SamPage:   TInputOptionWizardPage;   // Full vs core install
  NotesPage: TOutputMsgWizardPage;     // Model-weights info

// ─── Build the PowerShell flags string from user choices ─────────────────────
function GetPsFlags(Param: String): String;
var
  Flags: String;
begin
  Flags := '';

  // GPU page: index 2 = "CPU only"
  if GpuPage.Values[2] then
    Flags := Flags + ' -CPU';

  // SAM page: index 1 = "Core only (skip SAM)"
  if SamPage.Values[1] then
    Flags := Flags + ' -SkipSam';

  Result := Trim(Flags);
end;

// ─── Create custom pages ──────────────────────────────────────────────────────
procedure InitializeWizard();
begin

  // ── GPU / PyTorch page ───────────────────────────────────────────────────
  GpuPage := CreateInputOptionPage(
    wpSelectDir,
    'GPU Acceleration',
    'Select PyTorch GPU support',
    'PyTorch is required for road, building, and vegetation extraction. ' +
    'Choose the configuration that matches this machine:',
    True, False);
  GpuPage.Add('Auto-detect  (recommended)  — checks for NVIDIA GPU automatically');
  GpuPage.Add('NVIDIA GPU with CUDA 12.x  (RTX 20xx / 30xx / 40xx, A-series workstation)');
  GpuPage.Add('CPU only  (no NVIDIA GPU, or not sure)');
  GpuPage.Values[0] := True;

  // ── Feature-pack page ────────────────────────────────────────────────────
  SamPage := CreateInputOptionPage(
    GpuPage.ID,
    'Feature Pack',
    'Choose which features to install',
    'The full AI pack adds ~3 GB of packages and enables road, building, and ' +
    'vegetation extraction via SAM/OWLv2. Core classification always works without it.',
    True, False);
  SamPage.Add('Full install  (classification + AI-based road / building / vegetation extraction)');
  SamPage.Add('Core only  (classification only — faster install, ~3 GB smaller)');
  SamPage.Values[0] := True;

  // ── Model-weights information page ───────────────────────────────────────
  NotesPage := CreateOutputMsgPage(
    SamPage.ID,
    'AI Model Weights — Important Note',
    'The model weights are NOT included in this installer',
    'AI extraction features require HuggingFace model weights (~5 GB total).' +
    #13#10 +
    'These are too large to bundle and must be copied manually from the' +
    ' development machine.' +
    #13#10#13#10 +
    'On the development machine, copy:' +
    #13#10 +
    '  C:\Users\<name>\.cache\huggingface\hub\' +
    #13#10#13#10 +
    'Into the installation folder as:' +
    #13#10 +
    '  <install dir>\models\hf_cache\hub\' +
    #13#10#13#10 +
    'See STANDALONE_DEPLOYMENT.md (inside the install folder) for full details.' +
    #13#10#13#10 +
    'Click Next to start the installation.');

end;

// ─── Populate the "Ready to install" summary memo ────────────────────────────
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  GpuLabel, SamLabel: String;
begin
  if GpuPage.Values[0] then
    GpuLabel := 'Auto-detect'
  else if GpuPage.Values[1] then
    GpuLabel := 'NVIDIA CUDA 12.x'
  else
    GpuLabel := 'CPU only';

  if SamPage.Values[0] then
    SamLabel := 'Full install (classification + AI extraction)'
  else
    SamLabel := 'Core only (classification only)';

  Result :=
    MemoDirInfo + NewLine +
    NewLine + Space + 'PyTorch mode:   ' + GpuLabel +
    NewLine + Space + 'Feature pack:   ' + SamLabel +
    NewLine +
    NewLine + MemoTasksInfo;
end;

// ─── Warn user that install will open a console window ───────────────────────
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  MsgBox(
    'The installer will now open a console window to download and install ' +
    'Python packages (pip install).' + #13#10#13#10 +
    'This step may take 15–30 minutes depending on your internet speed.' + #13#10#13#10 +
    'Do NOT close the console window — it will close automatically when done.',
    mbInformation, MB_OK);
  Result := '';
end;
