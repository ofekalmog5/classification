import { useState, useEffect } from "react";
import { listDir } from "../api/client";

interface Entry { name: string; path: string; is_dir: boolean; }

interface Props {
  mode: "file" | "save";
  title: string;
  filter?: string;
  onSelect: (path: string | null) => void;
}

export default function FileBrowserModal({ mode, title, filter, onSelect }: Props) {
  const [dir, setDir] = useState("C:\\");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [saveName, setSaveName] = useState("calibration_profile.json");
  const [error, setError] = useState("");

  useEffect(() => {
    listDir(dir)
      .then((res) => {
        setEntries(res.entries);
        setError("");
      })
      .catch((e) => setError(String(e)));
  }, [dir]);

  const filtered = filter
    ? entries.filter((e) => e.is_dir || e.name.toLowerCase().endsWith(filter))
    : entries;

  const handleSelect = (entry: Entry) => {
    if (entry.is_dir) {
      setDir(entry.path);
    } else if (mode === "file") {
      onSelect(entry.path);
    }
  };

  const handleConfirm = () => {
    if (mode === "save") {
      const name = saveName.trim() || "profile.json";
      const full = dir.replace(/[\\/]$/, "") + "\\" + name;
      onSelect(full);
    }
  };

  const parentDir = dir.includes("\\") || dir.includes("/")
    ? dir.replace(/[\\/][^\\/]+[\\/]?$/, "") || dir
    : dir;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.7)" }}
    >
      <div style={{ background: "#1e1e1e", border: "1px solid #333", borderRadius: 8, width: 480, maxHeight: "70vh", display: "flex", flexDirection: "column", padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontWeight: 600, fontSize: 13 }}>{title}</span>
          <button onClick={() => onSelect(null)} style={{ background: "none", border: "none", color: "#aaa", fontSize: 18, cursor: "pointer" }}>×</button>
        </div>

        <div style={{ fontSize: 11, color: "#888", marginBottom: 6, wordBreak: "break-all" }}>{dir}</div>

        {error && <div style={{ fontSize: 11, color: "#f87171", marginBottom: 6 }}>{error}</div>}

        <div style={{ overflowY: "auto", flex: 1, border: "1px solid #333", borderRadius: 4, marginBottom: 8 }}>
          {dir !== parentDir && (
            <div
              onClick={() => setDir(parentDir)}
              style={{ padding: "5px 8px", fontSize: 12, cursor: "pointer", color: "#aaa" }}
            >
              .. (up)
            </div>
          )}
          {filtered.map((entry) => (
            <div
              key={entry.path}
              onClick={() => handleSelect(entry)}
              style={{
                padding: "5px 8px",
                fontSize: 12,
                cursor: "pointer",
                color: entry.is_dir ? "#7cb9e8" : "#e0e0e0",
                borderBottom: "1px solid #2a2a2a",
              }}
            >
              {entry.is_dir ? "📁 " : "📄 "}{entry.name}
            </div>
          ))}
        </div>

        {mode === "save" && (
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            style={{ width: "100%", padding: "4px 8px", fontSize: 12, background: "#111", border: "1px solid #444", borderRadius: 4, color: "#e0e0e0", marginBottom: 8 }}
            placeholder="filename.json"
          />
        )}

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={() => onSelect(null)} style={{ padding: "4px 12px", fontSize: 12, background: "#333", border: "none", borderRadius: 4, color: "#e0e0e0", cursor: "pointer" }}>
            Cancel
          </button>
          {mode === "save" && (
            <button onClick={handleConfirm} style={{ padding: "4px 12px", fontSize: 12, background: "#4a7c59", border: "none", borderRadius: 4, color: "#fff", cursor: "pointer" }}>
              Save Here
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
