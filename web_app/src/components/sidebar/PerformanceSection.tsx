import { useEffect } from "react";
import { useAppState, useAppDispatch } from "../../store";
import { suggestTileSize } from "../../api/client";
import type { TileSize } from "../../types";

const TILE_SIZES: TileSize[] = ["Auto", "256", "512", "1024", "2048", "4096"];

export default function PerformanceSection() {
  const state = useAppState();
  const { performance } = state;
  const dispatch = useAppDispatch();
  const accel = state.accelInfo;

  const set = (partial: Partial<typeof performance>) =>
    dispatch({ type: "SET_PERFORMANCE", settings: partial });

  // Resolve first raster-input file path from map layers (or rasterPath fallback)
  const rasterPath =
    state.mapLayers.find((l) => l.type === "raster-input")?.filePath ||
    state.rasterPath ||
    "";

  // Fetch suggested tile side from backend whenever the raster changes
  useEffect(() => {
    if (!rasterPath) {
      set({ suggestedTileSide: null });
      return;
    }
    let cancelled = false;
    suggestTileSize(rasterPath).then((side) => {
      if (!cancelled) set({ suggestedTileSide: side });
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasterPath]);

  const autoLabel = performance.suggestedTileSide
    ? `Auto (${performance.suggestedTileSide}×${performance.suggestedTileSide})`
    : "Auto";

  return (
    <SidebarSection title="Performance">
      {accel && (
        <div className={`flex items-center gap-1.5 py-0.5 text-xs ${accel.engine === "faiss-gpu" || accel.engine === "cuml" ? "text-emerald-400" : accel.engine === "faiss-cpu" ? "text-yellow-400" : "text-surface-500"}`}>
          <span>{accel.engine === "sklearn" ? "○" : "⬡"}</span>
          <span title={accel.info}>
            {accel.engine === "faiss-gpu" && `GPU+faiss: ${accel.info}`}
            {accel.engine === "faiss-cpu" && `faiss (CPU fast)`}
            {accel.engine === "cuml"      && `GPU+cuML: ${accel.info}`}
            {accel.engine === "sklearn"   && "CPU (sklearn)"}
          </span>
        </div>
      )}
      <ToggleRow
        label="Tile processing"
        id="perf-tiling"
        checked={performance.useTiling}
        onChange={(v) => set({ useTiling: v })}
      />

      <div className="flex items-center gap-2 mt-1">
        <label htmlFor="perf-tile-size" className="text-xs text-surface-400 w-20">Tile size</label>
        <select
          id="perf-tile-size"
          name="perf-tile-size"
          className="input flex-1 text-xs"
          value={performance.tileSize}
          onChange={(e) => set({ tileSize: e.target.value as TileSize })}
        >
          {TILE_SIZES.map((s) => {
            const safe = performance.suggestedTileSide;
            if (safe && s !== "Auto") {
              const side = parseInt(s);
              if (side > safe) return null;
            }
            return (
              <option key={s} value={s}>
                {s === "Auto" ? autoLabel : s}
              </option>
            );
          })}
        </select>
      </div>

      <NumberRow
        id="perf-tile-workers"
        label="Tile workers"
        value={performance.tileWorkers}
        min={1}
        max={64}
        onChange={(v) => set({ tileWorkers: v })}
      />

      <NumberRow
        id="perf-image-workers"
        label="Image workers"
        value={performance.imageWorkers}
        min={1}
        max={32}
        onChange={(v) => set({ imageWorkers: v })}
      />

      <ToggleRow
        id="perf-max-threads"
        label="Max threads"
        checked={performance.useMaxThreads}
        onChange={(v) => set({ useMaxThreads: v })}
      />
    </SidebarSection>
  );
}

/* ── Reusable mini-components ────────────────────────────────────── */

function ToggleRow({
  id,
  label,
  checked,
  onChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <label htmlFor={id} className="text-xs text-surface-400 cursor-pointer">{label}</label>
      <button
        id={id}
        name={id}
        className={`toggle-switch ${checked ? "active" : "inactive"}`}
        onClick={() => onChange(!checked)}
      >
        <span className="toggle-knob" />
      </button>
    </div>
  );
}

function NumberRow({
  id,
  label,
  value,
  min,
  max,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2 mt-0.5">
      <label htmlFor={id} className="text-xs text-surface-400 w-20">{label}</label>
      <input
        id={id}
        name={id}
        type="number"
        className="input flex-1 text-xs text-center"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value) || min)}
      />
    </div>
  );
}

function SidebarSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group">
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
