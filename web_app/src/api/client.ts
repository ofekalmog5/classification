/**
 * API client — talks to the FastAPI backend through the Vite proxy.
 *
 * All endpoints are prefixed with /api which gets rewritten to the
 * backend root by vite.config.ts.
 */
import type {
  ClassItem,
  VectorLayer,
  FeatureFlags,
  ClassifyResult,
  ProgressEvent,
} from "../types";

const BASE = "/api";

async function parseApiResponse<T>(r: Response): Promise<T> {
  const text = await r.text();
  let data: any = null;

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }

  if (!r.ok) {
    const message =
      (data && (data.message || data.error || data.detail)) ||
      text ||
      `HTTP ${r.status}`;
    throw new Error(message);
  }

  if (data === null) {
    throw new Error("Backend returned invalid JSON response");
  }

  return data as T;
}

/* ── Health ──────────────────────────────────────────────────────── */
export async function healthCheck(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/health`);
    return r.ok;
  } catch {
    return false;
  }
}

/* ── Classify Step 1 ─────────────────────────────────────────────── */
export interface Step1Params {
  rasterPath: string;
  classes: ClassItem[];
  featureFlags: FeatureFlags;
  outputPath?: string;
  exportFormat?: string;
  tileMode?: boolean;
  tileMaxPixels?: number;
  tileWorkers?: number;
  detectShadows?: boolean;
  maxThreads?: number | null;
  taskId?: string;
}

export async function runStep1(params: Step1Params): Promise<ClassifyResult> {
  const body = {
    rasterPath: params.rasterPath,
    classes: params.classes,
    smoothing: "none",
    featureFlags: params.featureFlags,
    outputPath: params.outputPath || null,
    exportFormat: params.exportFormat || "tif",
    tileMode: params.tileMode ?? false,
    tileMaxPixels: params.tileMaxPixels ?? 2048 * 2048,
    tileOverlap: 0,
    tileWorkers: params.tileWorkers ?? 4,
    detectShadows: params.detectShadows ?? false,
    maxThreads: params.maxThreads ?? null,
    taskId: params.taskId ?? null,
  };
  const r = await fetch(`${BASE}/classify-step1`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseApiResponse<ClassifyResult>(r);
}

/* ── Classify Step 2 ─────────────────────────────────────────────── */
export interface Step2Params {
  classificationPath: string;
  vectorLayers: VectorLayer[];
  classes: ClassItem[];
  outputPath?: string;
  tileMode?: boolean;
  tileMaxPixels?: number;
  tileWorkers?: number;
  maxThreads?: number | null;
  taskId?: string;
}

export async function runStep2(params: Step2Params): Promise<ClassifyResult> {
  const body = {
    classificationPath: params.classificationPath,
    vectorLayers: params.vectorLayers,
    classes: params.classes,
    outputPath: params.outputPath || null,
    tileMode: params.tileMode ?? false,
    tileMaxPixels: params.tileMaxPixels ?? 2048 * 2048,
    tileOverlap: 0,
    tileWorkers: params.tileWorkers ?? 4,
    maxThreads: params.maxThreads ?? null,
    taskId: params.taskId ?? null,
  };
  const r = await fetch(`${BASE}/classify-step2`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseApiResponse<ClassifyResult>(r);
}

/* ── Full pipeline ───────────────────────────────────────────────── */
export interface FullPipelineParams {
  rasterPath: string;
  classes: ClassItem[];
  vectorLayers: VectorLayer[];
  featureFlags: FeatureFlags;
  outputPath?: string;
  exportFormat?: string;
  tileMode?: boolean;
  tileMaxPixels?: number;
  tileWorkers?: number;
  detectShadows?: boolean;
  maxThreads?: number | null;
  taskId?: string;
}

export async function runFullPipeline(
  params: FullPipelineParams
): Promise<ClassifyResult> {
  const body = {
    rasterPath: params.rasterPath,
    classes: params.classes,
    vectorLayers: params.vectorLayers,
    smoothing: "none",
    featureFlags: params.featureFlags,
    outputPath: params.outputPath || null,
    exportFormat: params.exportFormat || "tif",
    tileMode: params.tileMode ?? false,
    tileMaxPixels: params.tileMaxPixels ?? 2048 * 2048,
    tileOverlap: 0,
    tileWorkers: params.tileWorkers ?? 4,
    detectShadows: params.detectShadows ?? false,
    maxThreads: params.maxThreads ?? null,
    taskId: params.taskId ?? null,
  };
  const r = await fetch(`${BASE}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseApiResponse<ClassifyResult>(r);
}

/* ── GPU info ────────────────────────────────────────────────────── */
export async function fetchGpuInfo(): Promise<{ available: boolean; info: string; engine: string } | null> {
  try {
    const r = await fetch(`${BASE}/gpu-info`);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

/* ── Suggest tile size ────────────────────────────────────────────── */
export async function suggestTileSize(rasterPath: string): Promise<number | null> {
  try {
    const r = await fetch(`${BASE}/suggest-tile-size`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rasterPath, workers: navigator.hardwareConcurrency ?? 4 }),
    });
    if (!r.ok) return null;
    const data = await r.json();
    return typeof data.side === "number" ? data.side : null;
  } catch {
    return null;
  }
}

/* ── Batch classify (shared model) ───────────────────────────────── */
export interface BatchParams {
  rasterPaths: string[];
  classes: ClassItem[];
  vectorLayers: VectorLayer[];
  featureFlags: FeatureFlags;
  outputPath?: string;
  exportFormat?: string;
  tileMode?: boolean;
  tileMaxPixels?: number;
  tileWorkers?: number;
  detectShadows?: boolean;
  maxThreads?: number | null;
  taskId?: string;
}

export interface BatchResult {
  status: "ok" | "error";
  message?: string;
  outputPaths?: string[];
  tileOutputs?: string[];
  results?: ClassifyResult[];
  errors?: Array<[string, string]>;
  meaMapping?: any[];
}

export async function runBatchClassify(params: BatchParams): Promise<BatchResult> {
  const body = {
    rasterPaths: params.rasterPaths,
    classes: params.classes,
    vectorLayers: params.vectorLayers,
    smoothing: "none",
    featureFlags: params.featureFlags,
    outputPath: params.outputPath || null,
    exportFormat: params.exportFormat || "tif",
    tileMode: params.tileMode ?? false,
    tileMaxPixels: params.tileMaxPixels ?? 2048 * 2048,
    tileOverlap: 0,
    tileWorkers: params.tileWorkers ?? 4,
    detectShadows: params.detectShadows ?? false,
    maxThreads: params.maxThreads ?? null,
    taskId: params.taskId ?? null,
  };
  const r = await fetch(`${BASE}/classify-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseApiResponse<BatchResult>(r);
}

/* ── Task ID helper ──────────────────────────────────────────────── */
export function generateTaskId(): string {
  return crypto.randomUUID();
}

/* ── Cancel a running task ───────────────────────────────────────── */
export async function cancelTask(taskId: string): Promise<void> {
  try {
    await fetch(`${BASE}/cancel/${taskId}`, { method: "POST" });
  } catch {
    /* ignore — backend may already have stopped */
  }
}

/* ── SSE progress stream ─────────────────────────────────────────── */
export function startProgressStream(
  taskId: string,
  onProgress: (evt: ProgressEvent) => void,
): () => void {
  const es = new EventSource(`${BASE}/progress/${taskId}`);
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as ProgressEvent;
      onProgress(data);
    } catch { /* ignore */ }
  };
  es.addEventListener("done", () => es.close());
  es.onerror = () => es.close();
  return () => es.close();
}

/* ── File browsing (used in electron / local mode) ───────────────── */
export async function browseFile(
  options?: { directory?: boolean; save?: boolean; filters?: string[] }
): Promise<string | null> {
  try {
    const r = await fetch(`${BASE}/browse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options ?? {}),
    });
    const data = await r.json();
    return data.path ?? null;
  } catch {
    return null;
  }
}

/* ── Recommend cluster count ─────────────────────────────────────── */
export interface RecommendParams {
  rasterPath: string;
  featureFlags: FeatureFlags;
  quick?: boolean;
}

export async function recommendClusters(
  params: RecommendParams
): Promise<{ recommended: number; detail?: string } | null> {
  try {
    const r = await fetch(`${BASE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

/* ── Scan folder for raster images ────────────────────────────────── */
export interface ScanFolderResult {
  folder: string;
  count: number;
  files: { name: string; path: string; relativePath: string }[];
}

export async function scanFolder(
  folderPath: string,
  extensions?: string[]
): Promise<ScanFolderResult | null> {
  try {
    const r = await fetch(`${BASE}/scan-folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folderPath, extensions: extensions ?? null }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

/* ── List raster tiles from a GeoTIFF (for map preview) ──────────── */
export async function getRasterInfo(
  filePath: string
): Promise<{ bounds: [[number, number], [number, number]]; crs: string } | null> {
  try {
    const r = await fetch(`${BASE}/raster-info`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filePath }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

/* ── List directory contents (for in-app file browser) ───────────── */
export interface DirEntry {
  name: string;
  type: "dir" | "file";
  size: number;
}

export interface ListDirResult {
  path: string;
  parent: string | null;
  entries: DirEntry[];
  error?: string;
}

export async function listDir(path?: string): Promise<ListDirResult | null> {
  try {
    const r = await fetch(`${BASE}/list-dir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path ?? null }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}
