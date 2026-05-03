import type { MeaClass } from "../types";

export const MEA_CLASSES: MeaClass[] = [
  { id: "class-1",  name: "BM_ASPHALT" },
  { id: "class-2",  name: "BM_CONCRETE" },
  { id: "class-3",  name: "BM_FOLIAGE" },
  { id: "class-4",  name: "BM_LAND_DRY_GRASS" },
  { id: "class-5",  name: "BM_LAND_GRASS" },
  { id: "class-6",  name: "BM_METAL" },
  { id: "class-7",  name: "BM_METAL_STEEL" },
  { id: "class-8",  name: "BM_PAINT_ASPHALT" },
  { id: "class-9",  name: "BM_ROCK" },
  { id: "class-10", name: "BM_SAND" },
  { id: "class-11", name: "BM_SOIL" },
  { id: "class-12", name: "BM_VEGETATION" },
  { id: "class-13", name: "BM_WATER" },
];

export const FACTORY_COLORS: Record<string, string> = {
  BM_ASPHALT:        "#2D2D30",
  BM_CONCRETE:       "#B4B4B4",
  BM_FOLIAGE:        "#006400",
  BM_LAND_DRY_GRASS: "#BDB76B",
  BM_LAND_GRASS:     "#7CFC00",
  BM_METAL:          "#A9ABB0",
  BM_METAL_STEEL:    "#708090",
  BM_PAINT_ASPHALT:  "#3C3F41",
  BM_ROCK:           "#827B73",
  BM_SAND:           "#EDC9AF",
  BM_SOIL:           "#654321",
  BM_VEGETATION:     "#228B22",
  BM_WATER:          "#1C6BA0",
};
