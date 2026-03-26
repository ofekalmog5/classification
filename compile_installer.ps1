# ============================================================
#  compile_installer.ps1
#  Run this on the DEV machine to produce ClassificationInstaller.exe
#  No external tools needed — uses .NET compiler built into Windows.
# ============================================================

$OUT = Join-Path $PSScriptRoot "ClassificationInstaller.exe"

$code = @'
using System;
using System.IO;
using System.Drawing;
using System.Windows.Forms;
using System.Diagnostics;
using System.Threading;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Reflection;

[assembly: AssemblyTitle("Classification Web App Installer")]
[assembly: AssemblyVersion("1.0.0.0")]

namespace CWAInstaller
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }

    class MainForm : Form
    {
        TextBox txtOffline, txtPython, txtInstall;
        Button  btnBrowseOffline, btnBrowsePy, btnBrowseInstall, btnAutoDetect, btnInstall;
        ProgressBar bar;
        RichTextBox log;
        Label lblStatus;

        public MainForm()
        {
            Build();
            AutoFill();
        }

        // ── UI ────────────────────────────────────────────────
        void Build()
        {
            Text            = "Classification Web App — Installer";
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox     = false;
            StartPosition   = FormStartPosition.CenterScreen;
            BackColor       = Color.White;

            // Header
            var hdr = new Panel { Dock = DockStyle.Top, Height = 68,
                                  BackColor = Color.FromArgb(0, 100, 200) };
            hdr.Controls.Add(Lbl("Classification Web App — Installer", 18, 10,
                                  new Font("Segoe UI", 13, FontStyle.Bold), Color.White));
            hdr.Controls.Add(Lbl("Offline installation — no internet required", 20, 44,
                                  new Font("Segoe UI", 9), Color.FromArgb(200, 225, 255)));
            Controls.Add(hdr);

            int y = 82;

            // Offline folder
            Controls.Add(BoldLbl("Offline installer folder:", 18, y));
            txtOffline = TB(18, y+22, 492); Controls.Add(txtOffline);
            btnBrowseOffline = Btn("Browse...", 518, y+21, 100);
            btnBrowseOffline.Click += (s,e) => PickFolder(txtOffline, "Select offline_installer folder");
            Controls.Add(btnBrowseOffline);
            y += 58;

            // Python exe
            Controls.Add(BoldLbl("Python 3.11.9 (python.exe):", 18, y));
            txtPython = TB(18, y+22, 390); Controls.Add(txtPython);
            btnBrowsePy = Btn("Browse...", 416, y+21, 90);
            btnBrowsePy.Click += (s,e) => {
                using (var d = new OpenFileDialog {
                    Title = "Select python.exe", Filter = "python.exe|python.exe|All|*.*" })
                    if (d.ShowDialog() == DialogResult.OK) txtPython.Text = d.FileName;
            };
            Controls.Add(btnBrowsePy);
            btnAutoDetect = Btn("Auto-detect", 514, y+21, 104);
            btnAutoDetect.Click += (s,e) => {
                string f = FindPython();
                if (f != null) { txtPython.Text = f; lblStatus.Text = "Found: " + f; }
                else lblStatus.Text = "Python 3.11 not found automatically — browse manually.";
            };
            Controls.Add(btnAutoDetect);
            y += 58;

            // Install dir
            Controls.Add(BoldLbl("Install to:", 18, y));
            txtInstall = TB(18, y+22, 492); txtInstall.Text = @"C:\ClassificationApp";
            Controls.Add(txtInstall);
            btnBrowseInstall = Btn("Browse...", 518, y+21, 100);
            btnBrowseInstall.Click += (s,e) => PickFolder(txtInstall, "Choose installation folder");
            Controls.Add(btnBrowseInstall);
            y += 56;

            // Status
            lblStatus = Lbl("Ready.", 18, y, new Font("Segoe UI", 8), Color.DimGray, 600);
            Controls.Add(lblStatus);
            y += 22;

            // Progress bar
            bar = new ProgressBar { Location = new Point(18, y),
                                    Size = new Size(600, 18), Value = 0 };
            Controls.Add(bar);
            y += 26;

            // Log
            log = new RichTextBox {
                Location   = new Point(18, y),
                Size       = new Size(600, 185),
                Font       = new Font("Consolas", 8),
                BackColor  = Color.FromArgb(12, 12, 12),
                ForeColor  = Color.FromArgb(180, 220, 180),
                ReadOnly   = true,
                ScrollBars = RichTextBoxScrollBars.Vertical,
                WordWrap   = false
            };
            Controls.Add(log);
            y += 193;

            // Install button
            btnInstall = new Button {
                Text      = "Install",
                Location  = new Point(488, y),
                Size      = new Size(130, 36),
                Font      = new Font("Segoe UI", 11, FontStyle.Bold),
                BackColor = Color.FromArgb(0, 100, 200),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            btnInstall.FlatAppearance.BorderSize = 0;
            btnInstall.Click += OnInstall;
            Controls.Add(btnInstall);

            ClientSize = new Size(636, y + 48);
        }

        // ── helpers ──────────────────────────────────────────
        Label Lbl(string t, int x, int y, Font f, Color c, int w = 500)
        {
            return new Label { Text=t, Location=new Point(x,y), Size=new Size(w,22), Font=f, ForeColor=c, BackColor=Color.Transparent };
        }
        Label BoldLbl(string t, int x, int y)
        {
            return new Label { Text=t, Location=new Point(x,y), Size=new Size(500,18),
                               Font=new Font("Segoe UI",9,FontStyle.Bold), ForeColor=Color.FromArgb(40,40,40) };
        }
        TextBox TB(int x, int y, int w)
        {
            return new TextBox { Location=new Point(x,y), Size=new Size(w,24), Font=new Font("Segoe UI",9) };
        }
        Button Btn(string t, int x, int y, int w)
        {
            return new Button { Text=t, Location=new Point(x,y), Size=new Size(w,26), FlatStyle=FlatStyle.System };
        }
        void PickFolder(TextBox tb, string desc)
        {
            using (var d = new FolderBrowserDialog { Description=desc,
                                                     SelectedPath=tb.Text,
                                                     ShowNewFolderButton=true })
                if (d.ShowDialog() == DialogResult.OK) tb.Text = d.SelectedPath;
        }

        // ── auto-fill ─────────────────────────────────────────
        void AutoFill()
        {
            string exe = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            foreach (var d in new[]{ Path.Combine(exe,"offline_installer"), exe })
                if (Directory.Exists(d) && (
                    Directory.Exists(Path.Combine(d,"offline_packages")) ||
                    Directory.Exists(Path.Combine(d,"app")) ||
                    File.Exists(Path.Combine(d,"Setup.bat"))
                ))
                { txtOffline.Text = d; break; }

            string py = FindPython();
            if (py != null) txtPython.Text = py;
        }

        string FindPython()
        {
            // 1. py launcher
            try {
                var p = Run0("py.exe", "-3.11 -c \"import sys;print(sys.executable)\"");
                string o = p.StandardOutput.ReadToEnd().Trim();
                p.WaitForExit();
                if (p.ExitCode == 0 && File.Exists(o)) return o;
            } catch {}

            // 2. 'python' in PATH
            try {
                var p = Run0("python", "--version");
                string v = p.StandardOutput.ReadToEnd() + p.StandardError.ReadToEnd();
                p.WaitForExit();
                if (v.Contains("3.11")) {
                    var p2 = Run0("python", "-c \"import sys;print(sys.executable)\"");
                    string o = p2.StandardOutput.ReadToEnd().Trim();
                    p2.WaitForExit();
                    if (File.Exists(o)) return o;
                }
            } catch {}

            // 3. common paths
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            foreach (var p in new[]{
                Path.Combine(local,"Python","pythoncore-3.11-64","python.exe"),
                Path.Combine(local,"Programs","Python","Python311","python.exe"),
                @"C:\Python311\python.exe",
                @"C:\Program Files\Python311\python.exe" })
                if (File.Exists(p)) return p;

            return null;
        }

        Process Run0(string exe, string args)
        {
            var psi = new ProcessStartInfo(exe, args) {
                UseShellExecute = false, RedirectStandardOutput = true,
                RedirectStandardError = true, CreateNoWindow = true };
            return Process.Start(psi);
        }

        // ── install ───────────────────────────────────────────
        void OnInstall(object s, EventArgs e)
        {
            if (!Directory.Exists(txtOffline.Text))
            { MB("Offline installer folder not found."); return; }
            if (!Directory.Exists(Path.Combine(txtOffline.Text,"offline_packages")))
            { MB("offline_packages\\ not found inside the selected folder.\nMake sure you selected the correct offline_installer folder."); return; }
            if (!File.Exists(txtPython.Text))
            { MB("python.exe not found.\nUse Auto-detect or browse to your Python 3.11 installation."); return; }
            if (string.IsNullOrWhiteSpace(txtInstall.Text))
            { MB("Choose an installation folder."); return; }

            foreach (Control c in new Control[]{ btnInstall,btnBrowseOffline,
                                                 btnBrowsePy,btnBrowseInstall,btnAutoDetect })
                c.Enabled = false;

            string offline = txtOffline.Text.TrimEnd('\\');
            string pyExe   = txtPython.Text;
            string inst    = txtInstall.Text.TrimEnd('\\');

            new Thread(() => {
                bool ok = false;
                try { ok = DoInstall(offline, pyExe, inst); }
                catch (Exception ex) { AppendLog("FATAL: " + ex.Message, Color.Red); }

                if (ok) {
                    Invoke(new Action(() =>
                        MessageBox.Show(
                            "Installation complete!\n\nInstalled to:\n" + inst +
                            "\n\nstart.bat         — launches backend + opens browser\nstart_backend.bat — backend only\n\nOr use the Desktop shortcut.",
                            "Done", MessageBoxButtons.OK, MessageBoxIcon.Information)));
                } else {
                    Invoke(new Action(() => btnInstall.Enabled = true));
                }
            }){ IsBackground = true }.Start();
        }

        bool DoInstall(string offline, string pyExe, string inst)
        {
            string pkgDir   = Path.Combine(offline, "offline_packages");
            string torchDir = Path.Combine(offline, "offline_packages_torch");
            string appSrc   = Path.Combine(offline, "app");
            string venvDir  = Path.Combine(inst, ".venv");
            string venvPy   = Path.Combine(venvDir, "Scripts", "python.exe");

            // 1 — install dir
            Status("Creating install directory..."); Pct(2);
            Log("\n=== [1/8] Install directory ===");
            Directory.CreateDirectory(inst);
            Log("  " + inst, Color.Green);

            // 2 — venv
            Status("Creating virtual environment..."); Pct(8);
            Log("\n=== [2/8] Python virtual environment ===");
            int rc = Exec(pyExe, "-m venv \"" + venvDir + "\"", "python -m venv");
            if (rc != 0 || !File.Exists(venvPy))
            { Log("ERROR: venv creation failed.", Color.Red); return false; }
            Log("  venv created.", Color.Green);

            // 3 — pip check / optional upgrade
            Status("Checking pip..."); Pct(12);
            Log("\n=== [3/8] pip ===");
            string[] pipWhl = Directory.GetFiles(pkgDir, "pip-*.whl");
            if (pipWhl.Length > 0)
            {
                Exec(venvPy, "-m pip install --no-index --find-links \"" + pkgDir + "\" --upgrade pip setuptools wheel", "upgrade pip");
                Log("  pip upgraded from cache.", Color.Gray);
            }
            else
            {
                Log("  pip.whl not in cache — using venv pip as-is.", Color.Gray);
            }

            // 4 — core packages
            Status("Installing packages (10–20 min)..."); Pct(18);
            Log("\n=== [4/8] Core packages ===");
            string reqFile = Path.Combine(appSrc, "requirements.txt");
            if (!File.Exists(reqFile)) reqFile = Path.Combine(appSrc, "backend", "requirements.txt");
            if (!File.Exists(reqFile)) reqFile = Path.Combine(offline, "requirements.txt");
            if (!File.Exists(reqFile))
                Log("  WARNING: requirements.txt not found — skipping.", Color.Yellow);
            else {
                string tmp = Path.Combine(Path.GetTempPath(), "cwa_req.txt");
                var keep = new List<string>();
                foreach (var l in File.ReadAllLines(reqFile)) {
                    if (Regex.IsMatch(l, @"^\s*#")) continue;
                    if (Regex.IsMatch(l, @"segment.geospatial|triton.windows")) continue;
                    if (l.Trim().Length == 0) continue;
                    keep.Add(l);
                }
                File.WriteAllLines(tmp, keep.ToArray());
                rc = Exec(venvPy,
                    "-m pip install --no-index --find-links \"" + pkgDir + "\" -r \"" + tmp + "\"",
                    "pip install requirements");
                if (rc != 0) Log("  WARNING: some packages failed (exit " + rc + ")", Color.Yellow);
                try { File.Delete(tmp); } catch {}
            }
            Log("  Core packages done.", Color.Green); Pct(52);

            // 5 — torch
            Status("Installing PyTorch..."); Pct(54);
            Log("\n=== [5/8] PyTorch ===");
            bool hasCuda = Directory.Exists(torchDir) && Directory.GetFiles(torchDir,"torch-*.whl").Length > 0;
            bool hasCpu  = Directory.GetFiles(pkgDir,"torch-*.whl").Length > 0;
            if (hasCuda) {
                Exec(venvPy,"-m pip install --no-index --find-links \""+torchDir+"\" torch torchvision","torch CUDA");
                Log("  PyTorch CUDA installed.", Color.Green);
            } else if (hasCpu) {
                Exec(venvPy,"-m pip install --no-index --find-links \""+pkgDir+"\" torch torchvision","torch CPU");
                Log("  PyTorch CPU installed.", Color.Green);
            } else Log("  No torch wheels — skipping.", Color.Yellow);
            Pct(66);

            // 6 — GPU packages (cupy / nvidia-cuda-*)
            Status("Installing GPU packages (cupy)...");
            Log("\n=== [6/8] GPU packages (cupy / nvidia-cuda-*) ===");
            string gpuDir = Path.Combine(offline, "offline_packages_gpu");
            if (Directory.Exists(gpuDir) && Directory.GetFiles(gpuDir, "*.whl").Length > 0) {
                string gpuReq = Path.Combine(appSrc, "requirements-gpu.txt");
                if (!File.Exists(gpuReq)) gpuReq = Path.Combine(appSrc, "backend", "requirements-gpu.txt");
                int gpuRc;
                if (File.Exists(gpuReq))
                    gpuRc = Exec(venvPy, "-m pip install --no-index --find-links \"" + gpuDir + "\" -r \"" + gpuReq + "\"", "pip install gpu packages");
                else
                    gpuRc = Exec(venvPy, "-m pip install --no-index --find-links \"" + gpuDir + "\" cupy-cuda12x", "pip install cupy");
                if (gpuRc != 0) Log("  WARNING: some GPU packages failed (exit " + gpuRc + ")", Color.Yellow);
                else Log("  GPU packages installed.", Color.Green);
            } else {
                Log("  offline_packages_gpu not found — skipping cupy.", Color.Gray);
            }
            Pct(72);

            // 7 — copy app files
            Status("Copying app files..."); Pct(74);
            Log("\n=== [7/8] App files ===");
            foreach (var sub in new[]{"backend","web_app","shared","models"})
                CopyDir(Path.Combine(appSrc,sub), Path.Combine(inst,sub));
            foreach (var f in new[]{"launcher.py","requirements.txt","requirements-gpu.txt","version.txt"}) {
                string src = Path.Combine(appSrc,f);
                if (File.Exists(src)) File.Copy(src, Path.Combine(inst,f), true);
            }

            // Patch main.py: inject /api prefix middleware if missing
            // (frontend calls /api/xxx; in dev Vite strips it, in production we must strip it here)
            string mainPy = Path.Combine(inst, "backend", "app", "main.py");
            if (File.Exists(mainPy)) {
                try {
                    string txt = File.ReadAllText(mainPy, System.Text.Encoding.UTF8);
                    if (!txt.Contains("_StripApiPrefix")) {
                        string mw =
                            "\r\n\r\n" +
                            "# /api prefix middleware: strips /api so standalone uvicorn matches Vite dev proxy\r\n" +
                            "from starlette.middleware.base import BaseHTTPMiddleware\r\n" +
                            "class _StripApiPrefix(BaseHTTPMiddleware):\r\n" +
                            "    async def dispatch(self, request, call_next):\r\n" +
                            "        p = request.scope.get(\"path\", \"\")\r\n" +
                            "        if p.startswith(\"/api\"):\r\n" +
                            "            s = p[4:] or \"/\"\r\n" +
                            "            request.scope[\"path\"] = s\r\n" +
                            "            request.scope[\"raw_path\"] = s.encode()\r\n" +
                            "        return await call_next(request)\r\n" +
                            "app.add_middleware(_StripApiPrefix)\r\n";
                        // Insert after CORS middleware closing paren
                        string mk1 = "allow_headers=[\"*\"]\r\n)";
                        string mk2 = "allow_headers=[\"*\"]\n)";
                        if (txt.Contains(mk1)) {
                            txt = txt.Replace(mk1, mk1 + mw);
                        } else if (txt.Contains(mk2)) {
                            txt = txt.Replace(mk2, mk2 + mw.Replace("\r\n", "\n"));
                        } else {
                            // Fallback: insert before static file section
                            int sfp = txt.IndexOf("_STATIC_DIR = ");
                            txt = sfp > 0
                                ? txt.Substring(0, sfp) + mw + txt.Substring(sfp)
                                : txt + mw;
                        }
                        File.WriteAllText(mainPy, txt, System.Text.Encoding.UTF8);
                        Log("  Patched main.py: /api middleware injected.", Color.Green);
                    } else {
                        Log("  main.py already has /api middleware.", Color.Gray);
                    }
                } catch (Exception ex) {
                    Log("  WARNING: could not patch main.py: " + ex.Message, Color.Yellow);
                }
            }

            // write start.bat — mirrors start_webapp.bat style: [1/2] backend, [2/2] browser
            File.WriteAllText(Path.Combine(inst,"start.bat"),
                "@echo off\r\n" +
                "cd /d \"%~dp0\"\r\n" +
                "echo Starting Classification Web App...\r\n" +
                "echo.\r\n" +
                "set HF_HUB_OFFLINE=1\r\n" +
                "set TRANSFORMERS_OFFLINE=1\r\n" +
                "set HF_HOME=%~dp0models\\hf_cache\r\n" +
                "\r\n" +
                ":: Kill any leftover process on port 8000\r\n" +
                "for /f \"tokens=5\" %%a in ('netstat -aon 2^>nul ^| findstr \":8000 \"') do taskkill /f /pid %%a >nul 2>&1\r\n" +
                "\r\n" +
                "echo [1/2] Starting backend server...\r\n" +
                "start \"Backend - FastAPI\" cmd /k \"cd /d %~dp0 && set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && set HF_HOME=%~dp0models\\hf_cache && .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1\"\r\n" +
                "\r\n" +
                ":: Poll /health until backend responds (max 60 sec)\r\n" +
                "set /a _t=0\r\n" +
                ":_wait\r\n" +
                "set /a _t+=1\r\n" +
                "if %_t% gtr 60 goto :_timeout\r\n" +
                "timeout /t 1 /nobreak >nul\r\n" +
                "powershell -NoProfile -Command \"try{if((Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 1 -EA Stop).StatusCode -eq 200){exit 0}}catch{exit 1}\" >nul 2>&1\r\n" +
                "if errorlevel 1 goto :_wait\r\n" +
                "\r\n" +
                "echo [2/2] Opening browser...\r\n" +
                "start http://127.0.0.1:8000\r\n" +
                "echo.\r\n" +
                "echo Both are running:\r\n" +
                "echo   Backend:  http://127.0.0.1:8000\r\n" +
                "echo   Browser:  http://127.0.0.1:8000\r\n" +
                "echo.\r\n" +
                "echo Close this window or press any key to exit.\r\n" +
                "pause >nul\r\n" +
                "goto :_end\r\n" +
                ":_timeout\r\n" +
                "echo.\r\n" +
                "echo WARNING: Backend did not respond after 60 seconds.\r\n" +
                "echo Check the 'Backend - FastAPI' window for errors.\r\n" +
                "echo Open http://127.0.0.1:8000 manually when ready.\r\n" +
                "pause\r\n" +
                ":_end\r\n");

            // write start_backend.bat — backend only, stays open so errors are visible
            File.WriteAllText(Path.Combine(inst,"start_backend.bat"),
                "@echo off\r\n" +
                "cd /d \"%~dp0\"\r\n" +
                "echo ============================================================\r\n" +
                "echo  Classification Web App - Backend Server\r\n" +
                "echo  http://127.0.0.1:8000   (close window to stop)\r\n" +
                "echo ============================================================\r\n" +
                "echo.\r\n" +
                "set HF_HUB_OFFLINE=1\r\n" +
                "set TRANSFORMERS_OFFLINE=1\r\n" +
                "set HF_HOME=%~dp0models\\hf_cache\r\n" +
                ".venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1\r\n" +
                "echo.\r\n" +
                "echo Backend stopped. Press any key to close.\r\n" +
                "pause >nul\r\n");

            Log("  Files copied.", Color.Green); Pct(88);

            // write check.bat — diagnoses missing packages
            File.WriteAllText(Path.Combine(inst,"check.bat"),
                "@echo off\r\ncd /d \"%~dp0\"\r\n" +
                "echo ============================================\r\n" +
                "echo  Installation Diagnostics\r\n" +
                "echo ============================================\r\n" +
                "echo.\r\n" +
                ".venv\\Scripts\\python.exe -c \"import fastapi\"  && echo [OK] fastapi   || echo [FAIL] fastapi\r\n" +
                ".venv\\Scripts\\python.exe -c \"import uvicorn\"  && echo [OK] uvicorn   || echo [FAIL] uvicorn\r\n" +
                ".venv\\Scripts\\python.exe -c \"import numpy\"    && echo [OK] numpy     || echo [FAIL] numpy\r\n" +
                ".venv\\Scripts\\python.exe -c \"import rasterio\" && echo [OK] rasterio  || echo [FAIL] rasterio\r\n" +
                ".venv\\Scripts\\python.exe -c \"import geopandas\"&& echo [OK] geopandas || echo [FAIL] geopandas\r\n" +
                ".venv\\Scripts\\python.exe -c \"import sklearn\"  && echo [OK] sklearn   || echo [FAIL] sklearn\r\n" +
                ".venv\\Scripts\\python.exe -c \"import cv2\"      && echo [OK] opencv    || echo [FAIL] opencv\r\n" +
                ".venv\\Scripts\\python.exe -c \"import shapely\"  && echo [OK] shapely   || echo [FAIL] shapely\r\n" +
                ".venv\\Scripts\\python.exe -c \"import faiss\"    && echo [OK] faiss     || echo [FAIL] faiss\r\n" +
                ".venv\\Scripts\\python.exe -c \"import torch\"    && echo [OK] torch     || echo [FAIL] torch\r\n" +
                ".venv\\Scripts\\python.exe -c \"import pynvml\"   && echo [OK] pynvml    || echo [FAIL] pynvml\r\n" +
                ".venv\\Scripts\\python.exe -c \"import cupy\"     && echo [OK] cupy      || echo [FAIL] cupy\r\n" +
                "echo.\r\n" +
                "echo Testing full backend import...\r\n" +
                ".venv\\Scripts\\python.exe -c \"import sys; sys.path.insert(0,'.'); import backend.app.main; print('[OK] backend loads OK')\" || echo [FAIL] backend has import error\r\n" +
                "echo.\r\npause\r\n");

            // 8 — shortcut
            Status("Creating shortcut..."); Pct(92);
            Log("\n=== [8/8] Desktop shortcut ===");
            MakeShortcut(
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                             "Classification Web App.lnk"),
                Path.Combine(inst,"start.bat"), inst);

            Pct(100);
            Log("\n=== COMPLETE ===", Color.LightGreen);
            Log("  Installed to: " + inst, Color.LightGreen);
            Log("  start.bat         — backend (new window) + opens browser", Color.LightGreen);
            Log("  start_backend.bat — backend only, no browser", Color.LightGreen);
            Log("\n  NOTE: copy SAM model weights to " + inst + "\\models\\hf_cache\\hub\\", Color.Yellow);
            Status("Installation complete!");
            return true;
        }

        int Exec(string exe, string args, string label)
        {
            AppendLog("  > " + label, Color.Cyan);
            var psi = new ProcessStartInfo(exe, args) {
                UseShellExecute=false, RedirectStandardOutput=true,
                RedirectStandardError=true, CreateNoWindow=true };
            var p = new Process { StartInfo=psi };
            p.OutputDataReceived += (s,e) => { if(e.Data!=null) AppendLog("    "+e.Data); };
            p.ErrorDataReceived  += (s,e) => { if(e.Data!=null) AppendLog("    "+e.Data, Color.Yellow); };
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            p.WaitForExit();
            return p.ExitCode;
        }

        void CopyDir(string src, string dst)
        {
            if (!Directory.Exists(src)) { AppendLog("  skip: " + src, Color.Gray); return; }
            foreach (var f in Directory.GetFiles(src,"*",SearchOption.AllDirectories)) {
                string rel = f.Substring(src.Length).TrimStart('\\','/');
                string d   = Path.Combine(dst, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(d));
                File.Copy(f, d, true);
            }
            AppendLog("  copied: " + src, Color.Gray);
        }

        void MakeShortcut(string lnk, string target, string workDir)
        {
            try {
                Type t  = Type.GetTypeFromProgID("WScript.Shell");
                object sh = Activator.CreateInstance(t);
                object sc = t.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, sh, new object[]{lnk});
                Type st = sc.GetType();
                st.InvokeMember("TargetPath",      BindingFlags.SetProperty, null, sc, new object[]{target});
                st.InvokeMember("WorkingDirectory",BindingFlags.SetProperty, null, sc, new object[]{workDir});
                st.InvokeMember("Save",            BindingFlags.InvokeMethod, null, sc, null);
                AppendLog("  shortcut created.", Color.Green);
            } catch(Exception ex) { AppendLog("  shortcut warning: "+ex.Message, Color.Yellow); }
        }

        void MB(string m) { MessageBox.Show(m,"Error",MessageBoxButtons.OK,MessageBoxIcon.Warning); }
        void Log(string m, Color? c=null) { AppendLog(m, c); }
        void AppendLog(string m, Color? c=null) {
            if(log.InvokeRequired){ log.Invoke(new Action(()=>AppendLog(m,c))); return; }
            log.SelectionColor = c ?? Color.FromArgb(180,220,180);
            log.AppendText("["+DateTime.Now.ToString("HH:mm:ss")+"] "+m+"\n");
            log.ScrollToCaret();
        }
        void Status(string m) {
            if(lblStatus.InvokeRequired){ lblStatus.Invoke(new Action(()=>Status(m))); return; }
            lblStatus.Text = m;
        }
        void Pct(int v) {
            if(bar.InvokeRequired){ bar.Invoke(new Action(()=>Pct(v))); return; }
            bar.Value = Math.Min(v,100);
        }
    }
}
'@

Write-Host "Compiling ClassificationInstaller.exe..."

Add-Type `
    -TypeDefinition $code `
    -ReferencedAssemblies "System.Windows.Forms","System.Drawing","System" `
    -OutputAssembly $OUT `
    -OutputType WindowsApplication

if (Test-Path $OUT) {
    Write-Host ""
    Write-Host "Done:  $OUT" -ForegroundColor Green
    Write-Host ""
    Write-Host "Copy ClassificationInstaller.exe to the target machine."
    Write-Host "Run it, point it to the offline_installer folder and python.exe."
} else {
    Write-Host "Compilation failed." -ForegroundColor Red
}
