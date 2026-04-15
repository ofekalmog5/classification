import { useState, useEffect, useCallback, useRef } from "react";
import { listDir, type DirEntry } from "../api/client";

const RASTER_EXTS = new Set([
  ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".img", ".asc", ".ecw", ".jp2",
]);
const VECTOR_EXTS = new Set([".shp", ".geojson", ".json", ".kml", ".gpkg"]);
const ALL_GIS_EXTS = new Set([...RASTER_EXTS, ...VECTOR_EXTS]);

interface Props {
  /** "file" = pick a single file, "folder" = pick a folder, "save" = pick/type a save path */
  mode: "file" | "folder" | "save";
  /** Called with the selected path, or null if cancelled */
  onSelect: (path: string | null) => void;
  /** Optional title override */
  title?: string;
  /** Optional file extension filter (e.g. [".tif",".shp"]) */
  extensions?: string[];
}

export default function FileBrowserModal({ mode, onSelect, title, extensions }: Props) {
  const [currentPath, setCurrentPath] = useState<string>("");
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [entries, setEntries] = useState<DirEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveName, setSaveName] = useState("");
  const [pathInput, setPathInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Filter extensions
  const allowedExts = extensions
    ? new Set(extensions.map((e) => e.toLowerCase()))
    : ALL_GIS_EXTS;

  const navigate = useCallback(async (path?: string) => {
    setLoading(true);
    setError(null);
    const result = await listDir(path);
    if (!result) {
      setError("Cannot read directory (is the backend running?)");
      setLoading(false);
      return;
    }
    if (result.error) {
      setError(result.error);
    }
    setCurrentPath(result.path);
    setParentPath(result.parent);
    setPathInput(result.path);
    setEntries(result.entries);
    setLoading(false);
  }, []);

  // Initial load — show drives / root
  useEffect(() => {
    navigate();
  }, [navigate]);

  const filteredEntries = entries.filter((e) => {
    if (e.type === "dir") return true;
    if (mode === "folder") return false; // folder mode hides files
    const ext = "." + e.name.split(".").pop()?.toLowerCase();
    return allowedExts.has(ext);
  });

  const handleEntryClick = (entry: DirEntry) => {
    if (entry.type === "dir") {
      const newPath = currentPath
        ? `${currentPath.replace(/[\\/]$/, "")}${sep()}${entry.name}`
        : entry.name; // drive root on Windows like "C:\\"
      navigate(newPath);
    } else {
      // File selected
      const fullPath = `${currentPath.replace(/[\\/]$/, "")}${sep()}${entry.name}`;
      onSelect(fullPath);
    }
  };

  const handleSelectFolder = () => {
    if (currentPath) onSelect(currentPath);
  };

  const handleSave = () => {
    if (!saveName.trim()) return;
    const fullPath = `${currentPath.replace(/[\\/]$/, "")}${sep()}${saveName.trim()}`;
    onSelect(fullPath);
  };

  const handleGoToPath = () => {
    if (pathInput.trim()) {
      navigate(pathInput.trim());
    }
  };

  const modalTitle =
    title ||
    (mode === "folder" ? "Select Folder" : mode === "save" ? "Save As" : "Select File");

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface-900 border border-surface-700 rounded-lg shadow-2xl w-[640px] max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-surface-700">
          <h2 className="text-sm font-semibold text-surface-100">{modalTitle}</h2>
          <button
            onClick={() => onSelect(null)}
            className="text-surface-400 hover:text-surface-200 text-lg leading-none"
          >
            ✕
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-surface-800 bg-surface-950">
          <button
            onClick={() => parentPath !== null && navigate(parentPath || undefined)}
            disabled={parentPath === null}
            className="btn-sm text-[11px] px-2 py-1 disabled:opacity-30"
            title="Go up"
          >
            ↑
          </button>
          <input
            ref={inputRef}
            type="text"
            className="input flex-1 text-xs py-1"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleGoToPath()}
            placeholder="Type a path and press Enter…"
          />
          <button onClick={handleGoToPath} className="btn-sm text-[11px] px-2 py-1">
            Go
          </button>
        </div>

        {/* File list */}
        <div className="flex-1 overflow-y-auto min-h-[200px] max-h-[400px]">
          {loading ? (
            <div className="flex items-center justify-center h-full text-surface-500 text-xs">
              Loading…
            </div>
          ) : error && entries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-red-400 text-xs px-4 text-center">
              {error}
            </div>
          ) : filteredEntries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-surface-500 text-xs">
              {mode === "folder" ? "No subdirectories" : "No matching files"}
            </div>
          ) : (
            <table className="w-full text-xs">
              <tbody>
                {filteredEntries.map((entry) => (
                  <tr
                    key={entry.name}
                    onClick={() => handleEntryClick(entry)}
                    className="cursor-pointer hover:bg-surface-800 transition-colors border-b border-surface-800/50"
                  >
                    <td className="px-3 py-1.5 w-6 text-center">
                      {entry.type === "dir" ? (
                        <span className="text-yellow-500">📁</span>
                      ) : (
                        <span className="text-surface-400">📄</span>
                      )}
                    </td>
                    <td className="py-1.5 text-surface-200 truncate max-w-[400px]">
                      {entry.name}
                    </td>
                    <td className="px-3 py-1.5 text-right text-surface-500 whitespace-nowrap">
                      {entry.type === "file" ? formatSize(entry.size) : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2 px-4 py-3 border-t border-surface-700 bg-surface-950">
          {mode === "save" && (
            <input
              type="text"
              className="input flex-1 text-xs py-1"
              placeholder="File name…"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
            />
          )}
          <div className="flex-1" />
          <button
            onClick={() => onSelect(null)}
            className="btn-sm text-[11px] px-3 py-1 opacity-70 hover:opacity-100"
          >
            Cancel
          </button>
          {mode === "folder" && (
            <button
              onClick={handleSelectFolder}
              disabled={!currentPath}
              className="btn-accent text-[11px] px-3 py-1 disabled:opacity-40"
            >
              Select This Folder
            </button>
          )}
          {mode === "save" && (
            <button
              onClick={handleSave}
              disabled={!saveName.trim()}
              className="btn-accent text-[11px] px-3 py-1 disabled:opacity-40"
            >
              Save
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Helpers ──────────────────────────────────────────────────────── */

function sep(): string {
  // Detect OS separator from known paths or default to backslash on Windows
  return navigator.platform?.startsWith("Win") ? "\\" : "/";
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
