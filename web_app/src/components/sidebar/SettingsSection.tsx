import { useState, useEffect } from "react";
import { getAppConfig, setAppConfig, type AppConfig } from "../../api/client";

export default function SettingsSection() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [sam3Path, setSam3Path] = useState("");
  const [hfCachePath, setHfCachePath] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // Load config on mount
  useEffect(() => {
    getAppConfig()
      .then((cfg) => {
        setConfig(cfg);
        setSam3Path(cfg.sam3_local_dir ?? "");
        setHfCachePath(cfg.hf_cache_dir ?? "");
      })
      .catch(() => {});
  }, []);

  const handleSave = async (updates: Partial<Omit<AppConfig, "configPath">>) => {
    setSaving(true);
    setMsg(null);
    try {
      const cfg = await setAppConfig(updates);
      setConfig(cfg);
      setSam3Path(cfg.sam3_local_dir ?? "");
      setHfCachePath(cfg.hf_cache_dir ?? "");
      setMsg("Saved");
      setTimeout(() => setMsg(null), 2000);
    } catch (e: any) {
      setMsg(`Error: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <details className="group">
      <summary className="flex items-center gap-2 cursor-pointer select-none py-1 text-xs font-semibold text-surface-400 uppercase tracking-wider hover:text-surface-200">
        <span className="transition-transform group-open:rotate-90">&#9654;</span>
        Settings
      </summary>
      <div className="mt-2 space-y-3 text-xs">
        {/* SAM3 Local Path */}
        <div>
          <label className="block text-surface-400 mb-1">SAM3 Local Path</label>
          <div className="flex gap-1">
            <input
              className="flex-1 bg-surface-800 text-surface-200 text-xs px-2 py-1 rounded border border-surface-600 focus:border-primary-500 outline-none"
              placeholder="Path to sam3-main folder..."
              value={sam3Path}
              onChange={(e) => setSam3Path(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave({ sam3_local_dir: sam3Path || null })}
            />
            <button
              className="px-2 py-1 bg-primary-600 hover:bg-primary-500 text-white rounded text-xs disabled:opacity-50"
              disabled={saving}
              onClick={() => handleSave({ sam3_local_dir: sam3Path || null })}
            >
              Set
            </button>
          </div>
          {config?.sam3_local_dir && (
            <p className="text-[10px] text-green-400 mt-0.5">
              Active: {config.sam3_local_dir}
            </p>
          )}
          {config && !config.sam3_local_dir && (
            <p className="text-[10px] text-surface-500 mt-0.5">
              Not set — will auto-detect from sibling folders
            </p>
          )}
        </div>

        {/* HuggingFace Cache Path */}
        <div>
          <label className="block text-surface-400 mb-1">HuggingFace Cache Dir</label>
          <div className="flex gap-1">
            <input
              className="flex-1 bg-surface-800 text-surface-200 text-xs px-2 py-1 rounded border border-surface-600 focus:border-primary-500 outline-none"
              placeholder="Custom HF cache path (leave empty for default)..."
              value={hfCachePath}
              onChange={(e) => setHfCachePath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave({ hf_cache_dir: hfCachePath || null })}
            />
            <button
              className="px-2 py-1 bg-primary-600 hover:bg-primary-500 text-white rounded text-xs disabled:opacity-50"
              disabled={saving}
              onClick={() => handleSave({ hf_cache_dir: hfCachePath || null })}
            >
              Set
            </button>
          </div>
          <p className="text-[10px] text-surface-500 mt-0.5">
            For standalone: copy the HF cache folder here and set the path
          </p>
        </div>

        {/* Offline Mode */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              className="accent-primary-500"
              checked={config?.offline_mode ?? false}
              onChange={(e) => handleSave({ offline_mode: e.target.checked })}
            />
            <span className="text-surface-300">Offline Mode</span>
          </label>
          <span className="text-[10px] text-surface-500">(no network, cached models only)</span>
        </div>

        {/* Status */}
        {msg && (
          <p className={`text-[10px] ${msg.startsWith("Error") ? "text-red-400" : "text-green-400"}`}>
            {msg}
          </p>
        )}

        {/* Config file location */}
        {config?.configPath && (
          <p className="text-[10px] text-surface-600 break-all">
            Config: {config.configPath}
          </p>
        )}
      </div>
    </details>
  );
}
