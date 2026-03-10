import { useEffect, useMemo, useRef, useState } from "react";
import { useAppState } from "../store";

export default function StatusBar() {
  const { running, progress, statusText } = useAppState();
  const isRunning = running !== "idle";
  const runStartedAtRef = useRef<number | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (isRunning && runStartedAtRef.current === null) {
      runStartedAtRef.current = Date.now();
    }
    if (!isRunning) {
      runStartedAtRef.current = null;
    }
  }, [isRunning]);

  // Tick every second while running so elapsed time updates
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  const timing = useMemo(() => {
    if (!isRunning || !runStartedAtRef.current) return null;

    // tick is in the deps to force recomputation every second
    void tick;

    const elapsedSec = Math.max(1, Math.floor((Date.now() - runStartedAtRef.current) / 1000));

    // Real progress from SSE — weighted phase-based percentage.
    if (progress && progress.total > 0 && progress.done > 0) {
      const pct = Math.min(99, Math.max(0, (progress.done / progress.total) * 100));
      // ETA: estimate remaining time from progress rate.
      // Use a smoothed rate to avoid jumps between fast/slow phases.
      const etaSec = pct > 0
        ? Math.max(0, Math.round(elapsedSec * (100 - pct) / pct))
        : 0;
      return {
        percent: pct,
        elapsedSec,
        etaSec,
        estimated: false,
      };
    }

    // Fallback when SSE hasn't sent events yet: indeterminate pulse.
    return {
      percent: -1,  // signals indeterminate
      elapsedSec,
      etaSec: 0,
      estimated: true,
    };
  }, [isRunning, progress, running, tick]);

  const isIndeterminate = timing !== null && timing.percent < 0;

  return (
    <footer className="flex items-center h-7 px-3 bg-surface-900 border-t border-surface-700 shrink-0 gap-3">
      {/* Status indicator */}
      <div className="flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            isRunning ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
          }`}
        />
        <span className="text-[11px] text-surface-400 truncate max-w-[520px]" title={statusText}>
          {statusText}
        </span>
      </div>

      {/* Progress bar + ETA */}
      {isRunning && timing && (
        <div className="flex items-center gap-2 flex-1 max-w-[460px]">
          <div className={`progress-bar flex-1${isIndeterminate ? " progress-indeterminate" : ""}`}>
            {!isIndeterminate && (
              <div
                className="progress-fill"
                style={{ width: `${Math.round(timing.percent)}%` }}
              />
            )}
            {isIndeterminate && (
              <div className="progress-fill" style={{ width: "100%" }} />
            )}
          </div>
          <span className="text-[10px] text-surface-500 whitespace-nowrap" title="Progress and ETA">
            {isIndeterminate
              ? `${formatSeconds(timing.elapsedSec)} elapsed`
              : `${Math.round(timing.percent)}% · ETA ${formatSeconds(timing.etaSec)} · ${formatSeconds(timing.elapsedSec)} elapsed`
            }
          </span>
        </div>
      )}

      {/* Phase label */}
      {isRunning && progress?.phase && (
        <span className="text-[10px] text-surface-500 truncate max-w-[200px]">
          {progress.phase}
        </span>
      )}

      <div className="flex-1" />

      {/* Running step label */}
      {isRunning && (
        <span className="text-[10px] text-amber-400/80 uppercase tracking-wider">
          {running}
        </span>
      )}
    </footer>
  );
}

function formatSeconds(totalSec: number): string {
  const sec = Math.max(0, Math.floor(totalSec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
