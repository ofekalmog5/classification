import { useState } from "react";
import { useAppState, useAppDispatch } from "../../store";
import type { VectorLayer } from "../../types";
import FileBrowserModal from "../FileBrowserModal";

export default function VectorsSection() {
  const { vectorLayers, classes } = useAppState();
  const dispatch = useAppDispatch();
  const [selectedClass, setSelectedClass] = useState("");
  const [showBrowser, setShowBrowser] = useState(false);

  const handleAddVector = () => {
    if (!classes.length) {
      alert("Define materials first.");
      return;
    }
    setShowBrowser(true);
  };

  const handleBrowserSelect = (path: string | null) => {
    setShowBrowser(false);
    if (!path) return;

    const classId = selectedClass || classes[0]?.id;
    if (!classId) return;

    const name = path.split(/[\\/]/).pop() || path;
    const layer: VectorLayer = {
      id: `vector-${Date.now()}`,
      name,
      filePath: path,
      classId,
    };
    dispatch({ type: "ADD_VECTOR", layer });
  };

  return (
    <SidebarSection title="Vector Overlay">
      <p className="text-[10px] text-surface-600 mb-1">
        Vectors will be drawn on result
      </p>

      <div className="flex gap-1 items-center">
        <select
          className="input flex-1 text-xs"
          value={selectedClass}
          onChange={(e) => setSelectedClass(e.target.value)}
        >
          {classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        <button className="btn-sm" onClick={handleAddVector}>
          Attach
        </button>
      </div>

      <div className="mt-2 max-h-36 overflow-y-auto rounded border border-surface-700 bg-surface-950">
        {vectorLayers.length === 0 && (
          <p className="text-xs text-surface-600 p-2 text-center">No vectors</p>
        )}
        {vectorLayers.map((l, i) => {
          const cls = classes.find((c) => c.id === l.classId);
          return (
            <div
              key={l.id}
              className="flex items-center gap-1.5 px-2 py-1 text-xs text-surface-300 hover:bg-surface-800 border-b border-surface-800 last:border-0"
            >
              <span className="text-surface-600">{i + 1}.</span>
              <span className="truncate flex-1">{l.name}</span>
              <span className="text-surface-500 text-[10px]">→ {cls?.name ?? "?"}</span>
              <button
                className="text-surface-600 hover:text-surface-300 text-[10px]"
                onClick={() => dispatch({ type: "MOVE_VECTOR", id: l.id, direction: -1 })}
              >
                ▲
              </button>
              <button
                className="text-surface-600 hover:text-surface-300 text-[10px]"
                onClick={() => dispatch({ type: "MOVE_VECTOR", id: l.id, direction: 1 })}
              >
                ▼
              </button>
              <button
                className="text-red-500/60 hover:text-red-400 text-[10px]"
                onClick={() => dispatch({ type: "REMOVE_VECTOR", id: l.id })}
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>

      {/* File browser modal for vectors */}
      {showBrowser && (
        <FileBrowserModal
          mode="file"
          onSelect={handleBrowserSelect}
          title="Select Shapefile"
          extensions={[".shp", ".geojson", ".json", ".kml", ".gpkg"]}
        />
      )}
    </SidebarSection>
  );
}

function SidebarSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group" open>
      <summary className="flex items-center cursor-pointer select-none py-2 px-1 text-xs font-semibold uppercase tracking-wider text-surface-400 hover:text-surface-200 transition-colors">
        <svg className="w-3 h-3 mr-1.5 transition-transform group-open:rotate-90" fill="currentColor" viewBox="0 0 20 20">
          <path d="M6 6l8 4-8 4V6z" />
        </svg>
        {title}
      </summary>
      <div className="pb-3 px-1 space-y-1">{children}</div>
    </details>
  );
}
