import { useAppState, useAppDispatch } from "../../store";
import { MEA_CLASSES, generatePalette } from "../../constants/mea";
import type { ClassItem } from "../../types";

export default function MaterialsSection() {
  const { classes, classCount } = useAppState();
  const dispatch = useAppDispatch();

  const applyCustom = () => {
    const count = Math.max(2, classCount);
    const palette = generatePalette(count);
    const items: ClassItem[] = Array.from({ length: count }, (_, i) => ({
      id: `class-${i + 1}`,
      name: `Class ${i + 1}`,
      color: palette[i],
    }));
    dispatch({ type: "SET_CLASSES", classes: items });
  };

  const setMEA = () => {
    dispatch({ type: "SET_CLASSES", classes: MEA_CLASSES });
    dispatch({ type: "SET_CLASS_COUNT", count: MEA_CLASSES.length });
  };

  return (
    <SidebarSection title="Materials">
      <div className="flex items-center gap-2">
        <label className="text-xs text-surface-400 whitespace-nowrap">Count</label>
        <input
          type="number"
          min={2}
          max={30}
          className="input w-16 text-center"
          value={classCount}
          onChange={(e) =>
            dispatch({ type: "SET_CLASS_COUNT", count: parseInt(e.target.value) || 2 })
          }
        />
        <button className="btn-sm" onClick={applyCustom}>
          Apply
        </button>
        <button className="btn-sm btn-accent" onClick={setMEA}>
          MEA Mode
        </button>
      </div>

      {/* Class list */}
      <div className="mt-2 max-h-44 overflow-y-auto rounded border border-surface-700 bg-surface-950">
        {classes.length === 0 && (
          <p className="text-xs text-surface-600 p-2 text-center">
            No materials defined
          </p>
        )}
        {classes.map((c) => (
          <div
            key={c.id}
            className="flex items-center gap-2 px-2 py-1 text-xs text-surface-300 hover:bg-surface-800 border-b border-surface-800 last:border-0"
          >
            <span
              className="w-3 h-3 rounded-sm shrink-0 border border-surface-600"
              style={{ backgroundColor: c.color }}
            />
            <span className="truncate">{c.name}</span>
            <span className="ml-auto text-surface-600 text-[10px]">{c.color}</span>
          </div>
        ))}
      </div>
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
