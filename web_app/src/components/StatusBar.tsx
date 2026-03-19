import { useEffect, useMemo, useRef, useState } from "react";
import { useAppState } from "../store";

// Sliding window ETA — tracks the last N (time, pct) samples to estimate
// the *current* throughput rate rather than the cumulative average.
const WINDOW_MAX_SAMPLES = 12;
const WINDOW_MAX_AGE_MS  = 45_000;   // discard samples older than 45 s
const EMA_ALPHA          = 0.25;     // exponential smoothing on rate (lower = smoother)

interface Sample { ts: number; pct: number }

export default function StatusBar() {
  const { running, progress, statusText } = useAppState();
  const isRunning = running !== "idle";

  const runStartedAtRef  = useRef<number | null>(null);
  const sampleWindowRef  = useRef<Sample[]>([]);
  const smoothedRateRef  = useRef<number | null>(null); // pct/sec
  const [tick, setTick]  = useState(0);

  // Reset on run start/stop
  useEffect(() => {
    if (isRunning && runStartedAtRef.current === null) {
      runStartedAtRef.current = Date.now();
    }
    if (!isRunning) {
      runStartedAtRef.current = null;
      sampleWindowRef.current = [];
      smoothedRateRef.current = null;
    }
  }, [isRunning]);

  // Tick every second so elapsed time updates
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  // Accumulate rate samples whenever a new progress event arrives
  useEffect(() => {
    if (!progress || progress.total <= 0 || progress.done <= 0) return;
    const pct = Math.min(99, Math.max(0, (progress.done / progress.total) * 100));
    const now  = Date.now();
    const win  = sampleWindowRef.current;

    // Only push if progress actually moved forward
    if (win.length === 0 || pct > win[win.length - 1].pct) {
      win.push({ ts: now, pct });

      // Prune old / excess samples
      const cutoff = now - WINDOW_MAX_AGE_MS;
      sampleWindowRef.current = win
        .filter((s) => s.ts >= cutoff)
        .slice(-WINDOW_MAX_SAMPLES);
    }

    // Re-estimate rate from the oldest vs newest sample in the window
    const w = sampleWindowRef.current;
    if (w.length >= 2) {
      const oldest = w[0];
      const newest = w[w.length - 1];
      const dPct   = newest.pct - oldest.pct;
      const dSec   = (newest.ts - oldest.ts) / 1000;
      if (dSec > 0.5 && dPct > 0) {
        const instantRate = dPct / dSec; // pct per second
        smoothedRateRef.current =
          smoothedRateRef.current === null
            ? instantRate
            : EMA_ALPHA * instantRate + (1 - EMA_ALPHA) * smoothedRateRef.current;
      }
    }
  }, [progress]);

  const timing = useMemo(() => {
    if (!isRunning || !runStartedAtRef.current) return null;
    void tick; // recompute every second

    const elapsedSec = Math.max(1, (Date.now() - runStartedAtRef.current) / 1000);
    const win = sampleWindowRef.current;

    if (win.length > 0 && progress && progress.total > 0) {
      const pct  = win[win.length - 1].pct;
      const rate = smoothedRateRef.current;

      let etaSec: number;
      if (rate && rate > 0) {
        // Primary: use current sliding-window rate
        etaSec = Math.max(0, (100 - pct) / rate);
      } else if (pct > 0) {
        // Fallback: linear from start (only before we have enough samples)
        etaSec = Math.max(0, elapsedSec * (100 - pct) / pct);
      } else {
        etaSec = 0;
      }

      return { percent: pct, elapsedSec, etaSec };
    }

    // No progress yet — indeterminate
    return { percent: -1, elapsedSec, etaSec: 0 };
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
        <div className="flex items-center gap-2 flex-1 max-w-[300px]">
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
