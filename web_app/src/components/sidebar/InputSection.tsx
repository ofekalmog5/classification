import { useState } from "react";
import { useAppState, useAppDispatch } from "../../store";
import { scanFolder } from "../../api/client";
import FileBrowserModal from "../FileBrowserModal";

type BrowserMode = null | { mode: "file" | "folder" | "save"; target: "input" | "output" };

export default function InputSection() {
  const { rasterPath, outputPath, lastResultPath, imageryMode, layerGroups } = useAppState();
  const dispatch = useAppDispatch();
  const [scanning, setScanning] = useState(false);
  const [browser, setBrowser] = useState<BrowserMode>(null);

  /** After file-browser selects a path */
  const handleBrowserSelect = async (path: string | null) => {
    const target = browser?.target;
    const mode = browser?.mode;
    setBrowser(null);
    if (!path || !target) return;

    if (target === "output") {
      dispatch({ type: "SET_OUTPUT_PATH", path });
      return;
    }

    // target === "input"
    dispatch({ type: "SET_RASTER_PATH", path });

    if (mode === "folder") {
      // Scan folder for images and add to map
      setScanning(true);
      dispatch({ type: "SET_STATUS", text: `Scanning ${path}…` });
      try {
        const result = await scanFolder(path);
        if (!result || result.count === 0) {
          dispatch({ type: "SET_STATUS", text: "No raster images found in folder." });
          setScanning(false);
          return;
        }
        // Use the first discovered raster as the active rasterPath for processing.
        // (Map still gets all files as layers.)
        if (result.files.length > 0) {
          dispatch({ type: "SET_RASTER_PATH", path: result.files[0].path });
        }

        // Create a group for this folder
        const folderName = path.split(/[\\/]/).pop() || "Input Images";
        const groupId = `group-input-${Date.now()}`;
        dispatch({
          type: "ADD_LAYER_GROUP",
          group: { id: groupId, name: folderName, visible: true, layerIds: [] },
        });

        const layerIds: string[] = [];
        for (let i = 0; i < result.files.length; i++) {
          const file = result.files[i];
          const layerId = `input-${Date.now()}-${i}`;
          dispatch({
            type: "ADD_MAP_LAYER",
            layer: {
              id: layerId,
              name: file.relativePath,
              type: "raster-input",
              filePath: file.path,
              visible: true,
              opacity: 1,
            },
          });
          layerIds.push(layerId);
        }
        dispatch({ type: "ADD_MANY_TO_GROUP", layerIds, groupId });

        const firstName = result.files[0]?.name || "";
        dispatch({
          type: "SET_STATUS",
          text: `Found ${result.count} image${result.count > 1 ? "s" : ""} in folder. Active raster: ${firstName}`,
        });
      } catch (e: any) {
        dispatch({
          type: "SET_STATUS",
          text: `Scan failed: ${e?.message || "check backend is running"}`,
        });
      }
      setScanning(false);
    } else {
      // Single file — add to map and place in "Input Images" group
      const layerId = `input-${Date.now()}`;
      dispatch({
        type: "ADD_MAP_LAYER",
        layer: {
          id: layerId,
          name: path.split(/[\\/]/).pop() || path,
          type: "raster-input",
          filePath: path,
          visible: true,
          opacity: 1,
        },
      });
      // Find or create the shared input group
      const existingGroup = layerGroups.find((g) => g.name === "Input Images");
      if (existingGroup) {
        dispatch({ type: "ADD_TO_GROUP", layerId, groupId: existingGroup.id });
      } else {
        const groupId = `group-input-${Date.now()}`;
        dispatch({
          type: "ADD_LAYER_GROUP",
          group: { id: groupId, name: "Input Images", visible: true, layerIds: [] },
        });
        dispatch({ type: "ADD_TO_GROUP", layerId, groupId });
      }
    }
  };

  return (
    <>
      <Section title="Input / Output">
        {/* Raster input */}
        <label className="label">Input raster</label>
        <div className="flex gap-1">
          <input
            type="text"
            className="input flex-1"
            placeholder="Path to raster file or folder…"
            value={rasterPath}
            onChange={(e) =>
              dispatch({ type: "SET_RASTER_PATH", path: e.target.value })
            }
          />
          <button
            className="btn-sm"
            onClick={() => setBrowser({ mode: "file", target: "input" })}
          >
            File
          </button>
          <button
            className="btn-sm"
            onClick={() => setBrowser({ mode: "folder", target: "input" })}
            disabled={scanning}
          >
            {scanning ? "…" : "Folder"}
          </button>
        </div>

        {/* Output */}
        <label className="label mt-2">Output</label>
        <div className="flex gap-1">
          <input
            type="text"
            className="input flex-1"
            placeholder="Output file or folder…"
            value={outputPath}
            onChange={(e) =>
              dispatch({ type: "SET_OUTPUT_PATH", path: e.target.value })
            }
          />
          <button
            className="btn-sm"
            onClick={() => setBrowser({ mode: "save", target: "output" })}
          >
            File
          </button>
          <button
            className="btn-sm"
            onClick={() => setBrowser({ mode: "folder", target: "output" })}
          >
            Folder
          </button>
        </div>
        {lastResultPath && (
          <p className="text-[10px] text-surface-500 mt-0.5 truncate" title={lastResultPath}>
            Last result: {lastResultPath}
          </p>
        )}

        {/* Imagery mode */}
        <label className="label mt-2">Imagery type</label>
        <div className="flex gap-3">
          {(["regular", "multispectral"] as const).map((m) => (
            <label key={m} className="flex items-center gap-1.5 text-xs text-surface-300 cursor-pointer">
              <input
                type="radio"
                name="imagery"
                className="accent-primary-500"
                checked={imageryMode === m}
                onChange={() => dispatch({ type: "SET_IMAGERY_MODE", mode: m })}
              />
              {m === "regular" ? "Regular (RGB)" : "Multispectral"}
            </label>
          ))}
        </div>
      </Section>

      {/* File browser modal */}
      {browser && (
        <FileBrowserModal
          mode={browser.mode}
          onSelect={handleBrowserSelect}
          title={
            browser.target === "output"
              ? browser.mode === "save"
                ? "Save Output File"
                : "Select Output Folder"
              : browser.mode === "folder"
                ? "Select Image Folder"
                : "Select Raster File"
          }
        />
      )}
    </>
  );
}

/* ── Helpers ──────────────────────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group" open>
      <summary className="flex items-center cursor-pointer select-none py-2 px-1 text-xs font-semibold uppercase tracking-wider text-surface-400 hover:text-surface-200 transition-colors">
        <svg
          className="w-3 h-3 mr-1.5 transition-transform group-open:rotate-90"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M6 6l8 4-8 4V6z" />
        </svg>
        {title}
      </summary>
      <div className="pb-3 px-1 space-y-1">{children}</div>
    </details>
  );
}
