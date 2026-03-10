import { useAppState, useAppDispatch } from "../../store";
import type { ExportFormat } from "../../types";

export default function ClassificationSection() {
  const { classification } = useAppState();
  const dispatch = useAppDispatch();

  const set = (partial: Partial<typeof classification>) =>
    dispatch({ type: "SET_CLASSIFICATION", settings: partial });

  return (
    <SidebarSection title="Classification">
      <ToggleRow
        label="Detect shadows"
        checked={classification.detectShadows}
        onChange={(v) => set({ detectShadows: v })}
      />
      <ToggleRow
        label="Share model (batch)"
        checked={classification.shareModel}
        onChange={(v) => set({ shareModel: v })}
      />
      <div className="flex items-center gap-2 mt-1">
        <span className="text-xs text-surface-400 w-20">Export format</span>
        <div className="flex gap-3">
          {(["tif", "img"] as ExportFormat[]).map((f) => (
            <label key={f} className="flex items-center gap-1.5 text-xs text-surface-300 cursor-pointer">
              <input
                type="radio"
                name="export-format"
                className="accent-primary-500"
                checked={classification.exportFormat === f}
                onChange={() => set({ exportFormat: f })}
              />
              {f.toUpperCase()}
            </label>
          ))}
        </div>
      </div>
    </SidebarSection>
  );
}

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
