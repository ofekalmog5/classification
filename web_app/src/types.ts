/* ── Shared domain types ──────────────────────────────────────── */

export interface ClassItem {
  id: string;
  name: string;
  color: string; // hex e.g. "#B4B4B4"
}

export interface VectorLayer {
  id: string;
  name: string;
  filePath: string;
  classId: string;
  overrideColor?: [number, number, number];
}

export interface FeatureFlags {
  spectral: boolean;
  texture: boolean;
  indices: boolean;
  colorIndices: boolean;
  entropy: boolean;
  morphCleanup: boolean;
}

export type ImageryMode = "regular" | "multispectral";
export type ExportFormat = "tif" | "img";
export type TileSize = "Auto" | "256" | "512" | "1024" | "2048" | "4096";

export interface PerformanceSettings {
  useTiling: boolean;
  tileSize: TileSize;
  tileWorkers: number;
  imageWorkers: number;
  useMaxThreads: boolean;
}

export interface ClassificationSettings {
  detectShadows: boolean;
  shareModel: boolean;
  exportFormat: ExportFormat;
}

/* ── Map layer types ────────────────────────────────────────────── */

export interface MapLayer {
  id: string;
  name: string;
  type: "raster-input" | "vector-overlay" | "classification-result";
  filePath: string;
  visible: boolean;
  opacity: number;
  groupId?: string;
}

export interface LayerGroup {
  id: string;
  name: string;
  visible: boolean;
  layerIds: string[];
}

/* ── API types ──────────────────────────────────────────────────── */

export interface MeaMapping {
  cluster: number;
  material: string;
  colorHex: string;
  colorRGB: [number, number, number];
}

export interface ClassifyResult {
  status: "ok" | "error";
  outputPath?: string;
  tileOutputs?: string[];
  message?: string;
  meaMapping?: MeaMapping[];
  statsTable?: string;
  saved?: string[];
  errors?: Array<[string, string]>;
}

export interface ProgressEvent {
  phase: string;
  done: number;
  total: number;
}

/* ── App state ──────────────────────────────────────────────────── */

export type PipelineStep =
  | "idle"
  | "step1"
  | "step2"
  | "step3"
  | "full"
  | "mea";

export interface AppState {
  rasterPath: string;
  outputPath: string;
  lastResultPath: string;
  imageryMode: ImageryMode;
  featureFlags: FeatureFlags;
  classes: ClassItem[];
  classCount: number;
  vectorLayers: VectorLayer[];
  performance: PerformanceSettings;
  classification: ClassificationSettings;
  // Map
  mapLayers: MapLayer[];
  layerGroups: LayerGroup[];
  // Run state
  running: PipelineStep;
  progress: ProgressEvent | null;
  statusText: string;
}
