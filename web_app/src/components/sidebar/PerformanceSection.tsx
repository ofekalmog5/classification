import { useAppState, useAppDispatch } from "../../store";
import type { TileSize } from "../../types";

const TILE_SIZES: TileSize[] = ["Auto", "256", "512", "1024", "2048", "4096"];

export default function PerformanceSection() {
  const { performance } = useAppState();
  const dispatch = useAppDispatch();

  const set = (partial: Partial<typeof performance>) =>
    dispatch({ type: "SET_PERFORMANCE", settings: partial });

  return (
    <SidebarSection title="Performance">
      <ToggleRow
        label="Tile processing"
        checked={performance.useTiling}
        onChange={(v) => set({ useTiling: v })}
      />

      <div className="flex items-center gap-2 mt-1">
        <label className="text-xs text-surface-400 w-20">Tile size</label>
        <select
          className="input flex-1 text-xs"
          value={performance.tileSize}
          onChange={(e) => set({ tileSize: e.target.value as TileSize })}
        >
          {TILE_SIZES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <NumberRow
        label="Tile workers"
        value={performance.tileWorkers}
        min={1}
        max={64}
        onChange={(v) => set({ tileWorkers: v })}
      />

      <NumberRow
        label="Image workers"
        value={performance.imageWorkers}
        min={1}
        max={32}
        onChange={(v) => set({ imageWorkers: v })}
      />

      <ToggleRow
        label="Max threads"
        checked={performance.useMaxThreads}
        onChange={(v) => set({ useMaxThreads: v })}
      />
    </SidebarSection>
  );
}

/* ── Reusable mini-components ────────────────────────────────────── */

function ToggleRow({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-xs text-surface-400">{label}</span>
      <button
        className={`toggle-switch ${checked ? "active" : "inactive"}`}
        onClick={() => onChange(!checked)}
      >
        <span className="toggle-knob" />
      </button>
    </div>
  );
}

function NumberRow({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2 mt-0.5">
      <label className="text-xs text-surface-400 w-20">{label}</label>
      <input
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
