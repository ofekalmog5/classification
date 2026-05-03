const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

// ─── Raster ──────────────────────────────────────────────────────────────────

export function getRasterInfo(path: string) {
  return post<{ width: number; height: number; bands: number; crs: string | null; bounds: number[] }>(
    "/raster-info",
    { path }
  );
}

export function getRasterAsPng(path: string): Promise<{ image_base64: string }> {
  return post("/raster-as-png", { path });
}

export function pickFile(filter = ""): Promise<{ path: string | null }> {
  return get(`/pick-file?filter=${encodeURIComponent(filter)}`);
}

export function pickSavePath(defaultName = "calibration_profile.json"): Promise<{ path: string | null }> {
  return get(`/pick-save-path?default_name=${encodeURIComponent(defaultName)}`);
}

export function listDir(directory: string) {
  return post<{ entries: Array<{ name: string; path: string; is_dir: boolean; size: number }> }>(
    "/list-dir",
    { directory }
  );
}

// ─── Sampling ────────────────────────────────────────────────────────────────

export interface SampleRegion {
  material: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MaterialSample {
  material: string;
  sample_count: number;
  mean_rgb: [number, number, number];
  std_rgb: [number, number, number];
  reference_color: string;
  reference_rgb: [number, number, number];
  tolerance_radius: number;
  anchor: [number, number, number];
}

export function samplePixels(rasterPath: string, regions: SampleRegion[]) {
  return post<{ status: string; samples: MaterialSample[] }>("/sample-pixels", { rasterPath, regions });
}

export function geoToRasterCoords(
  filePath: string,
  latLngTopLeft: [number, number],
  latLngBottomRight: [number, number]
) {
  return post<{ x: number; y: number; width: number; height: number }>(
    "/geo-to-raster",
    { filePath, latLngTopLeft, latLngBottomRight }
  );
}

// ─── Profile ─────────────────────────────────────────────────────────────────

export function getProfile() {
  return get<{ profile: import("../types").CalibrationProfile | null; profile_path: string; active: boolean }>(
    "/profile"
  );
}

export function getFactoryDefaults() {
  return get<{ defaults: import("../types").CalibrationProfile }>("/profile/factory-defaults");
}

export function saveProfile(data: {
  name: string;
  raster_path: string;
  material_overrides: Record<string, unknown>;
  bias_overrides?: Record<string, number>;
  frequency_prior_overrides?: Record<string, number>;
}) {
  return post<{ status: string; profile: import("../types").CalibrationProfile }>("/profile", data);
}

export function deleteProfile() {
  return fetch(`${BASE}/profile`, { method: "DELETE" }).then((r) => r.json());
}

export function importProfile(src_path: string) {
  return post<{ status: string; profile: import("../types").CalibrationProfile }>(
    "/profile/import",
    { src_path }
  );
}

export function exportProfile(dest_path: string) {
  return post<{ status: string; dest_path: string }>("/profile/export", { dest_path });
}
