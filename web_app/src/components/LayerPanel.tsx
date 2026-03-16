import { useState, useCallback, useRef, type DragEvent } from "react";
import { useAppState, useAppDispatch } from "../store";
import type { LayerGroup, MapLayer } from "../types";

export default function LayerPanel() {
  const { mapLayers, layerGroups } = useAppState();
  const dispatch = useAppDispatch();
  const [newGroupName, setNewGroupName] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const lastClickedRef = useRef<string | null>(null);

  const addGroup = () => {
    if (!newGroupName.trim()) return;
    const group: LayerGroup = {
      id: `group-${Date.now()}`,
      name: newGroupName.trim(),
      visible: true,
      layerIds: [],
    };
    dispatch({ type: "ADD_LAYER_GROUP", group });
    setNewGroupName("");
  };

  /** Select layers: plain = single, Ctrl = toggle-add one, Shift = range */
  const handleSelect = useCallback(
    (id: string, e: React.MouseEvent) => {
      // Build flat ordered list of all layer ids for range selection
      const allIds = mapLayers.map((l) => l.id);

      setSelectedIds((prev) => {
        if (e.shiftKey && lastClickedRef.current) {
          // Range select: from last-clicked to current
          const anchorIdx = allIds.indexOf(lastClickedRef.current);
          const targetIdx = allIds.indexOf(id);
          if (anchorIdx >= 0 && targetIdx >= 0) {
            const lo = Math.min(anchorIdx, targetIdx);
            const hi = Math.max(anchorIdx, targetIdx);
            const rangeIds = allIds.slice(lo, hi + 1);
            const next = new Set(prev);
            for (const rid of rangeIds) next.add(rid);
            return next;
          }
        }

        if (e.ctrlKey || e.metaKey) {
          // Toggle-add single
          const next = new Set(prev);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          lastClickedRef.current = id;
          return next;
        }

        // Plain click — select only this one
        lastClickedRef.current = id;
        return new Set([id]);
      });
    },
    [mapLayers],
  );

  /** Start dragging selected layers (or just the one being dragged) */
  const handleDragStart = useCallback(
    (layerId: string, e: DragEvent) => {
      const ids = selectedIds.has(layerId)
        ? Array.from(selectedIds)
        : [layerId];
      e.dataTransfer.setData("application/layer-ids", JSON.stringify(ids));
      e.dataTransfer.effectAllowed = "move";
    },
    [selectedIds],
  );

  // Layers not in any group
  const ungrouped = mapLayers.filter(
    (l) => !layerGroups.some((g) => g.layerIds.includes(l.id))
  );

  const hasSelected = selectedIds.size > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header + bulk visibility buttons */}
      <div className="px-3 py-2 border-b border-surface-700 flex items-center gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-surface-400 flex-1">
          Layers
        </h2>
        {mapLayers.length > 0 && (
          <div className="flex gap-1">
            {hasSelected ? (
              <>
                <button
                  className="text-[9px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-300 hover:bg-surface-600"
                  onClick={() =>
                    dispatch({ type: "SET_LAYERS_VISIBLE", ids: Array.from(selectedIds), visible: true })
                  }
                  title="Show selected layers"
                >
                  👁 Sel
                </button>
                <button
                  className="text-[9px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-300 hover:bg-surface-600"
                  onClick={() =>
                    dispatch({ type: "SET_LAYERS_VISIBLE", ids: Array.from(selectedIds), visible: false })
                  }
                  title="Hide selected layers"
                >
                  ◌ Sel
                </button>
              </>
            ) : (
              <>
                <button
                  className="text-[9px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-300 hover:bg-surface-600"
                  onClick={() => dispatch({ type: "SET_ALL_LAYERS_VISIBLE", visible: true })}
                  title="Show all layers"
                >
                  👁 All
                </button>
                <button
                  className="text-[9px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-300 hover:bg-surface-600"
                  onClick={() => dispatch({ type: "SET_ALL_LAYERS_VISIBLE", visible: false })}
                  title="Hide all layers"
                >
                  ◌ All
                </button>
              </>
            )}
            <button
              className="text-[9px] px-1.5 py-0.5 rounded bg-red-900/50 text-red-400 hover:bg-red-800/60 hover:text-red-300"
              onClick={() => {
                if (confirm("Remove all layers?")) {
                  dispatch({ type: "REMOVE_ALL_LAYERS" });
                  setSelectedIds(new Set());
                }
              }}
              title="Remove all layers and groups"
            >
              ✕ All
            </button>
          </div>
        )}
      </div>

      {/* Selection indicator */}
      {hasSelected && (
        <div className="px-3 py-1 bg-primary-900/30 border-b border-primary-700/40 flex items-center gap-2">
          <span className="text-[10px] text-primary-300">{selectedIds.size} selected</span>
          <button
            className="text-[9px] text-primary-400 hover:text-primary-200 underline"
            onClick={() => setSelectedIds(new Set())}
          >
            Clear
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {/* Layer groups */}
        {layerGroups.map((group) => (
          <GroupCard
            key={group.id}
            group={group}
            layers={mapLayers}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            onDragStart={handleDragStart}
          />
        ))}

        {/* Ungrouped layers */}
        {ungrouped.length > 0 && (
          <div className="space-y-0.5">
            {layerGroups.length > 0 && (
              <p className="text-[10px] text-surface-600 px-1 mb-1">Ungrouped</p>
            )}
            {ungrouped.map((layer) => (
              <LayerRow
                key={layer.id}
                layer={layer}
                selected={selectedIds.has(layer.id)}
                onSelect={handleSelect}
                onDragStart={handleDragStart}
              />
            ))}
          </div>
        )}

        {mapLayers.length === 0 && (
          <div className="text-center py-8 text-surface-600 text-xs">
            <p>No layers</p>
            <p className="text-[10px] mt-1">
              Run a classification or add inputs to see layers here
            </p>
          </div>
        )}
      </div>

      {/* Add group */}
      <div className="p-2 border-t border-surface-700">
        <div className="flex gap-1">
          <input
            type="text"
            className="input flex-1 text-xs"
            placeholder="New group name…"
            value={newGroupName}
            onChange={(e) => setNewGroupName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addGroup()}
          />
          <button className="btn-sm" onClick={addGroup}>
            + Group
          </button>
        </div>

        {/* Add input raster to map */}
        <button
          className="w-full mt-1.5 text-xs py-1 rounded bg-surface-800 hover:bg-surface-700 text-surface-400 transition-colors"
          onClick={() => {
            const path = window.prompt("Path to raster/result file:");
            if (!path) return;
            const name = path.split(/[\\/]/).pop() || "Layer";
            dispatch({
              type: "ADD_MAP_LAYER",
              layer: {
                id: `manual-${Date.now()}`,
                name,
                type: "raster-input",
                filePath: path,
                visible: true,
                opacity: 1,
              },
            });
          }}
        >
          + Add Layer to Map
        </button>
      </div>
    </div>
  );
}

/* ── Group card ──────────────────────────────────────────────────── */

function GroupCard({
  group,
  layers,
  selectedIds,
  onSelect,
  onDragStart,
}: {
  group: LayerGroup;
  layers: MapLayer[];
  selectedIds: Set<string>;
  onSelect: (id: string, e: React.MouseEvent) => void;
  onDragStart: (id: string, e: DragEvent) => void;
}) {
  const dispatch = useAppDispatch();
  const groupLayers = layers.filter((l) => group.layerIds.includes(l.id));
  const [expanded, setExpanded] = useState(true);
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    if (e.dataTransfer.types.includes("application/layer-ids")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      setDragOver(true);
    }
  };

  const handleDragLeave = () => setDragOver(false);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const raw = e.dataTransfer.getData("application/layer-ids");
    if (!raw) return;
    try {
      const ids: string[] = JSON.parse(raw);
      if (ids.length > 0) {
        dispatch({ type: "ADD_MANY_TO_GROUP", layerIds: ids, groupId: group.id });
      }
    } catch { /* ignore */ }
  };

  return (
    <div
      className={`rounded border bg-surface-800/50 transition-colors ${
        dragOver ? "border-primary-500 bg-primary-900/20" : "border-surface-700"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex items-center gap-2 px-2 py-1.5">
        <button
          className={`toggle-switch ${group.visible ? "active" : "inactive"}`}
          onClick={() => dispatch({ type: "TOGGLE_LAYER_GROUP", id: group.id })}
          style={{ transform: "scale(0.85)" }}
        >
          <span className="toggle-knob" />
        </button>
        <button
          className="flex-1 text-left text-xs font-medium text-surface-300 truncate"
          onClick={() => setExpanded((v) => !v)}
        >
          {group.name}
          <span className="text-surface-600 ml-1">({groupLayers.length})</span>
        </button>
        <button
          className="text-surface-600 hover:text-red-400 text-[10px]"
          onClick={() => dispatch({ type: "REMOVE_LAYER_GROUP", id: group.id })}
          title="Delete group"
        >
          ✕
        </button>
      </div>
      {expanded && groupLayers.length > 0 && (
        <div className="border-t border-surface-700 px-1 py-0.5">
          {groupLayers.map((l) => (
            <LayerRow
              key={l.id}
              layer={l}
              inGroup
              groupId={group.id}
              selected={selectedIds.has(l.id)}
              onSelect={onSelect}
              onDragStart={onDragStart}
            />
          ))}
        </div>
      )}
      {expanded && groupLayers.length === 0 && (
        <p className={`text-[10px] px-2 pb-1.5 ${dragOver ? "text-primary-300" : "text-surface-600"}`}>
          {dragOver ? "Drop layers here" : "Drag layers here or use + to add"}
        </p>
      )}
    </div>
  );
}

/* ── Layer row ───────────────────────────────────────────────────── */

function LayerRow({
  layer,
  inGroup,
  groupId,
  selected,
  onSelect,
  onDragStart,
}: {
  layer: MapLayer;
  inGroup?: boolean;
  groupId?: string;
  selected: boolean;
  onSelect: (id: string, e: React.MouseEvent) => void;
  onDragStart: (id: string, e: DragEvent) => void;
}) {
  const dispatch = useAppDispatch();
  const { layerGroups } = useAppState();

  const typeIcon = {
    "raster-input": "🗺️",
    "vector-overlay": "📐",
    "classification-result": "✨",
  }[layer.type];

  return (
    <div
      className={`flex items-center gap-1.5 px-1.5 py-1 rounded hover:bg-surface-700/50 group cursor-pointer transition-colors ${
        selected ? "bg-primary-900/40 ring-1 ring-primary-500/50" : ""
      }`}
      draggable
      onClick={(e) => {
        // Don't select when clicking controls
        if ((e.target as HTMLElement).closest("button, input, select")) return;
        onSelect(layer.id, e);
      }}
      onDragStart={(e) => onDragStart(layer.id, e as unknown as DragEvent)}
    >
      <button
        className={`toggle-switch ${layer.visible ? "active" : "inactive"}`}
        onClick={() => dispatch({ type: "TOGGLE_MAP_LAYER", id: layer.id })}
        style={{ transform: "scale(0.8)" }}
      >
        <span className="toggle-knob" />
      </button>
      <span className="text-[10px]">{typeIcon}</span>
      <span className="text-xs text-surface-300 truncate flex-1">{layer.name}</span>

      {/* Opacity slider */}
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={layer.opacity}
        onChange={(e) =>
          dispatch({
            type: "SET_LAYER_OPACITY",
            id: layer.id,
            opacity: parseFloat(e.target.value),
          })
        }
        className="w-12 h-1 accent-primary-500 opacity-0 group-hover:opacity-100 transition-opacity"
        title={`Opacity: ${Math.round(layer.opacity * 100)}%`}
      />

      {/* Add to group dropdown */}
      {!inGroup && layerGroups.length > 0 && (
        <select
          className="text-[10px] bg-transparent text-surface-600 opacity-0 group-hover:opacity-100 cursor-pointer w-0 group-hover:w-auto transition-all"
          value=""
          onChange={(e) => {
            if (e.target.value) {
              dispatch({
                type: "ADD_TO_GROUP",
                layerId: layer.id,
                groupId: e.target.value,
              });
            }
          }}
        >
          <option value="">📁</option>
          {layerGroups.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </select>
      )}

      {/* Remove from group */}
      {inGroup && groupId && (
        <button
          className="text-surface-600 hover:text-surface-400 text-[10px] opacity-0 group-hover:opacity-100"
          onClick={() =>
            dispatch({
              type: "REMOVE_FROM_GROUP",
              layerId: layer.id,
              groupId,
            })
          }
          title="Remove from group"
        >
          ↗
        </button>
      )}

      <button
        className="text-surface-600 hover:text-red-400 text-[10px] opacity-0 group-hover:opacity-100"
        onClick={() => dispatch({ type: "REMOVE_MAP_LAYER", id: layer.id })}
        title="Remove layer"
      >
        ✕
      </button>
    </div>
  );
}
