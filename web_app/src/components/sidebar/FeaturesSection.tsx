import { useAppState, useAppDispatch } from "../../store";
import type { FeatureFlags } from "../../types";

const FEATURE_DEFS: Array<{
  key: keyof FeatureFlags;
  label: string;
  tip?: string;
  requiresMulti?: boolean;
}> = [
  { key: "spectral",     label: "Spectral",      tip: "Per-band local mean" },
  { key: "texture",      label: "Texture",        tip: "Gray-level std-dev" },
  { key: "indices",      label: "NDVI",           tip: "Needs NIR band", requiresMulti: true },
  { key: "colorIndices", label: "VARI / HSV",     tip: "Visible color indices" },
  { key: "entropy",      label: "Entropy",        tip: "Local entropy (slow)" },
  { key: "morphCleanup", label: "Road cleanup",   tip: "Morphological post-process" },
];

export default function FeaturesSection() {
  const { featureFlags, imageryMode } = useAppState();
  const dispatch = useAppDispatch();

  const toggle = (key: keyof FeatureFlags) => {
    if (key === "indices" && imageryMode === "regular") return;
    dispatch({ type: "SET_FEATURE_FLAGS", flags: { [key]: !featureFlags[key] } });
  };

  return (
    <SidebarSection title="Algorithms">
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        {FEATURE_DEFS.map(({ key, label, tip, requiresMulti }) => {
          const disabled = requiresMulti && imageryMode === "regular";
          return (
            <label
              key={key}
              title={tip}
              className={`flex items-center gap-1.5 text-xs cursor-pointer ${
                disabled ? "text-surface-700 cursor-not-allowed" : "text-surface-300"
              }`}
            >
              <input
                type="checkbox"
                className="accent-primary-500"
                checked={featureFlags[key]}
                disabled={disabled}
                onChange={() => toggle(key)}
              />
              {label}
            </label>
          );
        })}
      </div>
    </SidebarSection>
  );
}

function SidebarSection({ title, children }: { title: string; children: React.ReactNode }) {
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
