import { useState, useEffect } from "react";
import { useAppState, useAppDispatch } from "../store";
import { MEA_CLASS_ENTRIES, FACTORY_COLORS } from "../constants/mea";
import {
  samplePixels,
  saveProfile,
  deleteProfile,
  importProfile,
  exportProfile,
  getProfile,
  pickFile,
  pickSavePath,
} from "../api/client";
import type { CalibrationProfile } from "../types";

const S: Record<string, React.CSSProperties> = {
  sidebar: { width: 260, background: "#1a1a1a", borderRight: "1px solid #2a2a2a", display: "flex", flexDirection: "column", overflow: "hidden" },
  header: { padding: "10px 12px", borderBottom: "1px solid #2a2a2a", fontSize: 13, fontWeight: 700, color: "#e0e0e0" },
  section: { padding: "10px 12px", borderBottom: "1px solid #222" },
  label: { fontSize: 11, color: "#888", marginBottom: 4 },
  btn: { padding: "4px 10px", fontSize: 11, borderRadius: 4, border: "none", cursor: "pointer", background: "#333", color: "#e0e0e0" },
  btnAccent: { background: "#4a7c59", color: "#fff" },
  btnDanger: { background: "#7c2b2b", color: "#fff" },
  input: { width: "100%", padding: "4px 8px", fontSize: 11, background: "#111", border: "1px solid #444", borderRadius: 4, color: "#e0e0e0" },
  matRow: { display: "flex", alignItems: "center", gap: 6, padding: "4px 8px", fontSize: 11, cursor: "pointer", borderBottom: "1px solid #1e1e1e" },
  matRowActive: { background: "#2a3a2a" },
  swatch: { width: 12, height: 12, borderRadius: 2, border: "1px solid #444", flexShrink: 0 },
  error: { fontSize: 11, color: "#f87171", padding: "0 12px 6px" },
};

export default function CalibrationSidebar() {
  const { calibration, rasterPath, profileColors } = useAppState();
  const dispatch = useAppDispatch();
  const [sampling, setSampling] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const swatches = Object.values(calibration.swatches);
  const calibratedCount = swatches.filter((s) => s.computedMeanRgb).length;
  const hasRegions = swatches.some((s) => s.regions.length > 0);

  // Load existing profile on mount
  useEffect(() => {
    getProfile()
      .then((res) => {
        if (res.active && res.profile) {
          dispatch({ type: "CAL_SET_PROFILE", profile: res.profile, profilePath: res.profile_path });
        }
      })
      .catch(() => {});
  }, []);

  /* ── Open raster ─────────────────────────────────────────────────── */
  const handleOpenRaster = async () => {
    setError(null);
    const { path } = await pickFile(".tif");
    if (path) dispatch({ type: "SET_RASTER_PATH", path });
  };

  /* ── Start session ───────────────────────────────────────────────── */
  const handleStart = () => {
    if (!rasterPath) {
      setError("Open an orthophoto first.");
      return;
    }
    setError(null);
    dispatch({ type: "CAL_OPEN", rasterPath });
  };

  /* ── Sample ──────────────────────────────────────────────────────── */
  const handleSampleAll = async () => {
    if (!hasRegions) return;
    setSampling(true);
    setError(null);
    try {
      const regions: { material: string; x: number; y: number; width: number; height: number }[] = [];
      for (const swatch of swatches) {
        for (const reg of swatch.regions) {
          regions.push({ material: swatch.material, ...reg });
        }
      }
      const result = await samplePixels(calibration.rasterPath, regions);
      for (const s of result.samples) {
        dispatch({
          type: "CAL_SET_SWATCH_RESULT",
          material: s.material,
          meanRgb: s.reference_rgb,
          stdRgb: s.std_rgb,
          sampleCount: s.sample_count,
        });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Sampling failed");
    }
    setSampling(false);
  };

  /* ── Save ────────────────────────────────────────────────────────── */
  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const material_overrides: CalibrationProfile["material_overrides"] = {};
      for (const [name, swatch] of Object.entries(calibration.swatches)) {
        if (!swatch.computedMeanRgb) continue;
        const [r, g, b] = swatch.computedMeanRgb;
        const hex = `#${r.toString(16).padStart(2, "0").toUpperCase()}${g.toString(16).padStart(2, "0").toUpperCase()}${b.toString(16).padStart(2, "0").toUpperCase()}`;
        const std = swatch.computedStdRgb ?? [0, 0, 0];
        material_overrides[name] = {
          reference_color: hex,
          reference_rgb: swatch.computedMeanRgb,
          anchors: [swatch.computedMeanRgb],
          sample_count: swatch.sampleCount,
          sample_std_rgb: std,
        };
      }
      const res = await saveProfile({
        name: calibration.profileName || "Calibration Profile",
        raster_path: calibration.rasterPath,
        material_overrides,
      });
      dispatch({ type: "CAL_SET_PROFILE", profile: res.profile, profilePath: null });
      dispatch({ type: "CAL_CLOSE" });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
    setSaving(false);
  };

  /* ── Delete ──────────────────────────────────────────────────────── */
  const handleDelete = async () => {
    if (!window.confirm("Delete the active calibration profile?")) return;
    setError(null);
    try {
      await deleteProfile();
      dispatch({ type: "CAL_SET_PROFILE", profile: null, profilePath: null });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  /* ── Import ──────────────────────────────────────────────────────── */
  const handleImport = async () => {
    setError(null);
    const { path } = await pickFile(".json");
    if (!path) return;
    try {
      const res = await importProfile(path);
      dispatch({ type: "CAL_SET_PROFILE", profile: res.profile, profilePath: path });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    }
  };

  /* ── Export ──────────────────────────────────────────────────────── */
  const handleExport = async () => {
    setError(null);
    const { path } = await pickSavePath("calibration_profile.json");
    if (!path) return;
    try {
      await exportProfile(path);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Export failed");
    }
  };

  return (
    <div style={S.sidebar}>
      <div style={S.header}>MEA Calibration Tool</div>

      {/* Raster input */}
      <div style={S.section}>
        <div style={S.label}>Orthophoto</div>
        {rasterPath ? (
          <div style={{ fontSize: 11, color: "#9fce9f", marginBottom: 6, wordBreak: "break-all" }}>
            {rasterPath.split(/[\\/]/).pop()}
          </div>
        ) : (
          <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>No file selected</div>
        )}
        <button style={S.btn} onClick={handleOpenRaster}>Open Ortho…</button>
      </div>

      {error && <div style={S.error}>{error}</div>}

      {/* Profile status */}
      {!calibration.active && (
        <div style={S.section}>
          <div style={S.label}>Active Profile</div>
          {calibration.savedProfile ? (
            <>
              <div style={{ fontSize: 11, color: "#e0e0e0", marginBottom: 4, fontWeight: 600 }}>
                {calibration.savedProfile.name}
              </div>
              <div style={{ fontSize: 10, color: "#888", marginBottom: 6 }}>
                {Object.keys(calibration.savedProfile.material_overrides).length} materials calibrated
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <button style={{ ...S.btn, ...S.btnAccent }} onClick={handleStart}>Edit</button>
                <button style={S.btn} onClick={handleExport}>Export</button>
                <button style={S.btn} onClick={handleImport}>Import</button>
                <button style={{ ...S.btn, ...S.btnDanger }} onClick={handleDelete}>Delete</button>
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>Using factory defaults</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <button style={{ ...S.btn, ...S.btnAccent }} onClick={handleStart}>New Session</button>
                <button style={S.btn} onClick={handleImport}>Load Profile</button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Active calibration session */}
      {calibration.active && (
        <>
          <div style={S.section}>
            <div style={S.label}>Profile Name</div>
            <input
              style={S.input}
              value={calibration.profileName}
              onChange={(e) => dispatch({ type: "CAL_SET_NAME", name: e.target.value })}
              placeholder="Profile name…"
            />
          </div>

          <div style={{ flex: 1, overflowY: "auto" }}>
            {MEA_CLASS_ENTRIES.map((cls) => {
              const swatch = calibration.swatches[cls.name];
              const isActive = calibration.activeMaterial === cls.name;
              const regionCount = swatch?.regions.length ?? 0;
              const sampled = swatch?.computedMeanRgb;
              const color = sampled
                ? `rgb(${sampled.join(",")})`
                : (profileColors[cls.name] ?? FACTORY_COLORS[cls.name] ?? "#aaa");
              const absorbsHint = cls.subAbsorbs.length
                ? `absorbs ${cls.subAbsorbs.map((n) => n.replace("BM_", "")).join(", ")}`
                : "";

              return (
                <div
                  key={cls.name}
                  onClick={() =>
                    dispatch({ type: "CAL_SET_ACTIVE_MATERIAL", material: isActive ? null : cls.name })
                  }
                  style={{ ...S.matRow, ...(isActive ? S.matRowActive : {}) }}
                  title={absorbsHint || undefined}
                >
                  <span style={{ ...S.swatch, background: color }} />
                  <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                    <span style={{ fontFamily: "monospace", fontSize: 10, color: isActive ? "#9fce9f" : "#ccc" }}>
                      {cls.name.replace("BM_", "")}
                    </span>
                    {absorbsHint && (
                      <span style={{ fontSize: 8, color: "#666", marginTop: 1 }}>{absorbsHint}</span>
                    )}
                  </div>
                  {regionCount > 0 && (
                    <span style={{ fontSize: 10, color: "#888" }}>{regionCount}r</span>
                  )}
                  {sampled && <span style={{ fontSize: 10, color: "#9fce9f" }}>✓</span>}
                  {regionCount > 0 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); dispatch({ type: "CAL_CLEAR_REGIONS", material: cls.name }); }}
                      style={{ background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: 13, padding: 0 }}
                      title="Clear regions"
                    >×</button>
                  )}
                </div>
              );
            })}
          </div>

          {calibration.activeMaterial && (
            <div style={{ padding: "6px 12px", fontSize: 11, color: "#9fce9f", borderTop: "1px solid #222" }}>
              Draw rectangles over <strong>{calibration.activeMaterial}</strong>
            </div>
          )}

          <div style={{ padding: "10px 12px", display: "flex", gap: 6, flexWrap: "wrap", borderTop: "1px solid #2a2a2a" }}>
            <button
              style={{ ...S.btn, ...S.btnAccent }}
              onClick={handleSampleAll}
              disabled={sampling || !hasRegions}
            >
              {sampling ? "Sampling…" : `Sample (${calibratedCount}/${MEA_CLASS_ENTRIES.length})`}
            </button>
            <button
              style={S.btn}
              onClick={handleSave}
              disabled={saving || calibratedCount === 0}
            >
              {saving ? "Saving…" : "Save Profile"}
            </button>
            <button style={S.btn} onClick={() => dispatch({ type: "CAL_CLOSE" })}>
              Cancel
            </button>
          </div>
        </>
      )}

    </div>
  );
}
