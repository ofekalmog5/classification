import type { ClassItem, MeaClassEntry } from "../types";

/**
 * MEA 6-material schema. Each entry carries metadata used by the calibration UI
 * and the SAM3-first pipeline (see backend/app/core.py MEA_CLASSES).
 *
 *   - source:        "mask" pixels come from SAM3/shapefile, "kmeans" from RGB clustering
 *   - subAbsorbs:    legacy 13-material names that this class now covers
 *   - compositeName: display name in the Composite_Material_Table XML output
 */
export const MEA_CLASS_ENTRIES: MeaClassEntry[] = [
  { id: "class-1", name: "BM_ASPHALT",    color: "#2D2D30", compositeName: "ASPHALT",       source: "mask",   subAbsorbs: ["BM_PAINT_ASPHALT"] },
  { id: "class-2", name: "BM_CONCRETE",   color: "#B4B4B4", compositeName: "CONCRETE",      source: "mask",   subAbsorbs: ["BM_ROCK", "BM_METAL", "BM_METAL_STEEL"] },
  { id: "class-3", name: "BM_VEGETATION", color: "#228B22", compositeName: "GENVEGETATION", source: "kmeans", subAbsorbs: ["BM_FOLIAGE", "BM_LAND_GRASS", "BM_LAND_DRY_GRASS"] },
  { id: "class-4", name: "BM_WATER",      color: "#1C6BA0", compositeName: "WATER",         source: "kmeans", subAbsorbs: [] },
  { id: "class-5", name: "BM_SAND",       color: "#EDC9AF", compositeName: "SAND",          source: "kmeans", subAbsorbs: [] },
  { id: "class-6", name: "BM_SOIL",       color: "#654321", compositeName: "SOIL",          source: "kmeans", subAbsorbs: [] },
];

/**
 * Network-payload view of MEA_CLASS_ENTRIES — what gets sent to /classify.
 * The backend's ClassItem Pydantic model only needs id, name, color.
 */
export const MEA_CLASSES: ClassItem[] = MEA_CLASS_ENTRIES.map(({ id, name, color }) => ({ id, name, color }));

/** Factory hex color per BM_* name — used by the calibration tool sidebar. */
export const FACTORY_COLORS: Record<string, string> = Object.fromEntries(
  MEA_CLASS_ENTRIES.map(({ name, color }) => [name, color])
);

/** Default palette for N custom classes */
const BASE_PALETTE = [
  "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
  "#00FFFF", "#FF8000", "#8000FF", "#00FF80", "#FF0080",
  "#0080FF", "#80FF00", "#FF4500", "#1E90FF", "#32CD32",
  "#FF1493", "#FFD700", "#4B0082", "#00CED1", "#FF6347",
  "#9400D3", "#00FA9A", "#FF69B4", "#1E90FF", "#ADFF2F",
  "#DC143C", "#00BFFF", "#7FFF00", "#FF00FF", "#20B2AA",
];

export function generatePalette(count: number): string[] {
  if (count <= BASE_PALETTE.length) return BASE_PALETTE.slice(0, count);
  const colors = [...BASE_PALETTE];
  while (colors.length < count) {
    const hue = (colors.length * 0.618033988749895) % 1.0;
    const sat = colors.length % 2 === 0 ? 0.9 : 0.7;
    const val = Math.floor(colors.length / 2) % 2 === 0 ? 0.95 : 0.75;
    const [r, g, b] = hsvToRgb(hue, sat, val);
    colors.push(
      `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
    );
  }
  return colors;
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r: number, g: number, b: number;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
