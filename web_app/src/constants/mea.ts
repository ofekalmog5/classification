import type { ClassItem } from "../types";

/**
 * MEA predefined material classes matching backend/app/core.py MEA_CLASSES.
 */
export const MEA_CLASSES: ClassItem[] = [
  { id: "class-1",  name: "BM_ASPHALT",        color: "#2D2D30" },
  { id: "class-2",  name: "BM_CONCRETE",       color: "#B4B4B4" },
  { id: "class-3",  name: "BM_FOLIAGE",        color: "#006400" },
  { id: "class-4",  name: "BM_LAND_DRY_GRASS", color: "#BDB76B" },
  { id: "class-5",  name: "BM_LAND_GRASS",     color: "#7CFC00" },
  { id: "class-6",  name: "BM_METAL",          color: "#A9ABB0" },
  { id: "class-7",  name: "BM_METAL_STEEL",    color: "#708090" },
  { id: "class-8",  name: "BM_PAINT_ASPHALT",  color: "#3C3F41" },
  { id: "class-9",  name: "BM_ROCK",           color: "#827B73" },
  { id: "class-10", name: "BM_SAND",           color: "#EDC9AF" },
  { id: "class-11", name: "BM_SOIL",           color: "#654321" },
  { id: "class-12", name: "BM_VEGETATION",     color: "#228B22" },
  { id: "class-13", name: "BM_WATER",          color: "#1C6BA0" },
];

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
