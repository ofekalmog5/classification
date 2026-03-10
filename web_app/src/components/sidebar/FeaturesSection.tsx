import { useAppState, useAppDispatch } from "../../store";

export default function FeaturesSection() {
  const { featureFlags, imageryMode } = useAppState();
  const dispatch = useAppDispatch();

  const toggle = (key: keyof typeof featureFlags) => {
    if (key === "indices" && imageryMode === "regular") return;
    dispatch({ type: "SET_FEATURE_FLAGS", flags: { [key]: !featureFlags[key] } });
  };

  return (
    <SidebarSection title="Features">
      <div className="flex gap-4">
        {(["spectral", "texture", "indices"] as const).map((k) => {
          const disabled = k === "indices" && imageryMode === "regular";
          return (
            <label
              key={k}
              className={`flex items-center gap-1.5 text-xs cursor-pointer ${
                disabled ? "text-surface-700 cursor-not-allowed" : "text-surface-300"
              }`}
            >
              <input
                type="checkbox"
                className="accent-primary-500"
                checked={featureFlags[k]}
                disabled={disabled}
                onChange={() => toggle(k)}
              />
              {k.charAt(0).toUpperCase() + k.slice(1)}
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
