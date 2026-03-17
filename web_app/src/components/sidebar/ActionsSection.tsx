import { useCallback, useRef } from "react";
import { useAppState, useAppDispatch } from "../../store";
import {
  runStep1,
  runStep2,
  runFullPipeline,
  runBatchClassify,
  generateTaskId,
  startProgressStream,
  cancelTask,
} from "../../api/client";
import type { BatchResult } from "../../api/client";
import { MEA_CLASSES } from "../../constants/mea";

function formatElapsed(ms: number): string {
  const sec = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export default function ActionsSection() {
  const state = useAppState();
  const dispatch = useAppDispatch();
  const isRunning = state.running !== "idle";
  const runStartRef = useRef<number>(0);
  const activeTaskIdRef = useRef<string | null>(null);

  const handleCancel = useCallback(async () => {
    const tid = activeTaskIdRef.current;
    if (!tid) return;
    dispatch({ type: "SET_STATUS", text: "Cancelling…" });
    await cancelTask(tid);
  }, [dispatch]);

  // Short engine label shown in the status bar when a run starts
  const engineLabel = state.accelInfo
    ? ` · ${
        state.accelInfo.engine === "faiss-gpu"  ? "GPU"
        : state.accelInfo.engine === "cupy"     ? "GPU (cupy)"
        : state.accelInfo.engine === "cuml"     ? "GPU (cuML)"
        : state.accelInfo.engine === "faiss-cpu" ? "faiss-cpu"
        : "CPU"
      }`
    : "";

  const commonParams = useCallback(() => {
    const maxThreads = state.performance.useMaxThreads
      ? navigator.hardwareConcurrency ?? null
      : null;
    return {
      classes: state.classes,
      featureFlags: state.featureFlags,
      outputPath: state.outputPath || undefined,
      exportFormat: state.classification.exportFormat,
      tileMode: state.performance.useTiling,
      tileMaxPixels: tileSizeToPixels(state.performance.tileSize, state.performance.suggestedTileSide),
      tileWorkers: state.performance.tileWorkers,
      detectShadows: state.classification.detectShadows,
      maxThreads,
    };
  }, [state]);

  /** Collect all raster-input file paths from map layers */
  const getRasterFiles = useCallback((): string[] => {
    const files = state.mapLayers
      .filter((l) => l.type === "raster-input")
      .map((l) => l.filePath)
      .filter((p) => p && !looksLikeDirectoryPath(p));
    // deduplicate
    return [...new Set(files)];
  }, [state.mapLayers]);

  /** Create a results group and add output layers to it */
  const addResultsToGroup = useCallback(
    (label: string, outputPaths: string[]) => {
      if (outputPaths.length === 0) return;
      const groupId = `result-group-${Date.now()}`;
      dispatch({
        type: "ADD_LAYER_GROUP",
        group: { id: groupId, name: label, visible: true, layerIds: [] },
      });
      const layerIds: string[] = [];
      for (const p of outputPaths) {
        const name = p.split(/[\\/]/).pop() || "Result";
        const id = `result-${Date.now()}-${name}`;
        layerIds.push(id);
        dispatch({
          type: "ADD_MAP_LAYER",
          layer: {
            id,
            name,
            type: "classification-result",
            filePath: p,
            visible: true,
            opacity: 0.85,
          },
        });
      }
      if (layerIds.length > 0) {
        dispatch({ type: "ADD_MANY_TO_GROUP", layerIds, groupId });
      }
    },
    [dispatch],
  );

  /** Process a BatchResult — add layers to a group even on partial success */
  const handleBatchResult = useCallback(
    (result: BatchResult, label: string) => {
      const elapsed = formatElapsed(Date.now() - runStartRef.current);
      const paths: string[] = result.outputPaths ?? [];
      const errCount = result.errors?.length ?? 0;

      // Log individual errors for debugging
      if (errCount > 0) {
        console.warn(`[${label}] ${errCount} error(s):`, result.errors);
      }

      if (result.status === "ok") {
        const suffix = errCount > 0 ? ` (${errCount} warnings)` : "";
        dispatch({ type: "SET_STATUS", text: `${label} complete ✓ (${elapsed})${suffix}` });
      } else if (paths.length > 0) {
        // Partial success: some files produced output
        dispatch({
          type: "SET_STATUS",
          text: `${label} partial: ${result.message ?? "Some files failed"} (${elapsed})`,
        });
      } else {
        dispatch({
          type: "SET_STATUS",
          text: `${label} failed: ${result.message ?? "Unknown error"}`,
        });
      }

      // Always add whatever output paths we got (even on partial failure)
      if (paths.length > 0) {
        dispatch({ type: "SET_LAST_RESULT_PATH", path: paths[paths.length - 1] });
        addResultsToGroup(label, paths);
      }
    },
    [dispatch, addResultsToGroup],
  );

  const handleStep1 = async () => {
    const rasterFiles = getRasterFiles();
    if (rasterFiles.length === 0) {
      if (!state.rasterPath) return alert("Select a raster image first.");
      if (looksLikeDirectoryPath(state.rasterPath)) {
        return alert("Input raster must be a file, not a folder. Choose a raster file (.tif/.img/.jpg...).");
      }
      rasterFiles.push(state.rasterPath);
    }
    if (!state.classes.length) return alert("Define materials first.");
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step1" });
    dispatch({ type: "SET_PROGRESS", progress: null });

    const params = commonParams();

    // Use batch endpoint (shared model) when multiple files
    if (rasterFiles.length > 1) {
      dispatch({ type: "SET_STATUS", text: `Step 1: Training shared model on ${rasterFiles.length} files…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const result = await runBatchClassify({
          rasterPaths: rasterFiles,
          vectorLayers: [],
          taskId,
          ...params,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Step 1 cancelled" });
        } else {
          handleBatchResult(result, "Step 1 (batch)");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `Step 1 batch error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    } else {
      // Single file — use original endpoint
      const file = rasterFiles[0];
      const fileName = file.split(/[\\/]/).pop() || file;
      dispatch({ type: "SET_STATUS", text: `Step 1: Classifying ${fileName}…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const result = await runStep1({ rasterPath: file, taskId, ...params });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Step 1 cancelled" });
        } else {
          handleSingleResult(result, "Step 1");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `Step 1 error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    }
    dispatch({ type: "SET_RUNNING", step: "idle" });
  };

  const handleStep2 = async () => {
    if (!state.lastResultPath) return alert("Run Step 1 first to produce a classification file.");
    if (!state.vectorLayers.length) return alert("Add at least one vector layer.");
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step2" });
    dispatch({ type: "SET_STATUS", text: "Running Step 2: Vector rasterization…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    activeTaskIdRef.current = taskId;
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const result = await runStep2({
        classificationPath: state.lastResultPath,
        vectorLayers: state.vectorLayers,
        classes: state.classes,
        outputPath: state.outputPath || undefined,
        taskId,
        tileMode: state.performance.useTiling,
        tileMaxPixels: tileSizeToPixels(state.performance.tileSize, state.performance.suggestedTileSide),
        tileWorkers: state.performance.tileWorkers,
        maxThreads: state.performance.useMaxThreads
          ? navigator.hardwareConcurrency ?? null
          : null,
      });
      if ((result as any).status === "cancelled") {
        dispatch({ type: "SET_STATUS", text: "Step 2 cancelled" });
      } else {
        handleSingleResult(result, "Step 2");
      }
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Step 2 error: ${e.message}` });
    } finally {
      activeTaskIdRef.current = null;
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleFull = async () => {
    const rasterFiles = getRasterFiles();
    if (rasterFiles.length === 0) {
      if (!state.rasterPath) return alert("Select a raster image first.");
      if (looksLikeDirectoryPath(state.rasterPath)) {
        return alert("Input raster must be a file, not a folder. Choose a raster file (.tif/.img/.jpg...).");
      }
      rasterFiles.push(state.rasterPath);
    }
    if (!state.classes.length) return alert("Define materials first.");
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "full" });
    dispatch({ type: "SET_PROGRESS", progress: null });

    const params = commonParams();

    if (rasterFiles.length > 1) {
      dispatch({ type: "SET_STATUS", text: `Full Pipeline: Training shared model on ${rasterFiles.length} files…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const result = await runBatchClassify({
          rasterPaths: rasterFiles,
          vectorLayers: state.vectorLayers,
          taskId,
          ...params,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Full Pipeline cancelled" });
        } else {
          handleBatchResult(result, "Full Pipeline (batch)");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `Full pipeline batch error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    } else {
      const file = rasterFiles[0];
      const fileName = file.split(/[\\/]/).pop() || file;
      dispatch({ type: "SET_STATUS", text: `Full Pipeline: ${fileName}…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const result = await runFullPipeline({
          rasterPath: file,
          vectorLayers: state.vectorLayers,
          taskId,
          ...params,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Full Pipeline cancelled" });
        } else {
          handleSingleResult(result, "Full Pipeline");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `Full pipeline error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    }
    dispatch({ type: "SET_RUNNING", step: "idle" });
  };

  const handleMEA = async () => {
    const rasterFiles = getRasterFiles();
    if (rasterFiles.length === 0) {
      if (!state.rasterPath) return alert("Select a raster image first.");
      if (looksLikeDirectoryPath(state.rasterPath)) {
        return alert("MEA requires raster file(s). You selected a folder path.");
      }
      rasterFiles.push(state.rasterPath);
    }
    dispatch({ type: "SET_CLASSES", classes: MEA_CLASSES });
    dispatch({ type: "SET_CLASS_COUNT", count: MEA_CLASSES.length });
    dispatch({ type: "SET_CLASSIFICATION", settings: { exportFormat: "img" } });
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "mea" });
    dispatch({ type: "SET_PROGRESS", progress: null });

    if (rasterFiles.length > 1) {
      dispatch({ type: "SET_STATUS", text: `MEA: Training shared model on ${rasterFiles.length} files…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const maxThreads = state.performance.useMaxThreads
          ? navigator.hardwareConcurrency ?? null
          : null;
        const result = await runBatchClassify({
          rasterPaths: rasterFiles,
          classes: MEA_CLASSES,
          vectorLayers: state.vectorLayers,
          featureFlags: state.featureFlags,
          outputPath: state.outputPath || undefined,
          exportFormat: "img",
          tileMode: state.performance.useTiling,
          tileMaxPixels: tileSizeToPixels(state.performance.tileSize, state.performance.suggestedTileSide),
          tileWorkers: state.performance.tileWorkers,
          detectShadows: state.classification.detectShadows,
          maxThreads,
          taskId,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "MEA cancelled" });
        } else {
          handleBatchResult(result, "MEA (batch)");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `MEA batch error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    } else {
      const file = rasterFiles[0];
      const fileName = file.split(/[\\/]/).pop() || file;
      dispatch({ type: "SET_STATUS", text: `MEA: ${fileName}…${engineLabel}` });
      const taskId = generateTaskId();
      activeTaskIdRef.current = taskId;
      const stopProgress = startProgressStream(taskId, (evt) => {
        dispatch({ type: "SET_PROGRESS", progress: evt });
      });
      try {
        const params = {
          rasterPath: file,
          classes: MEA_CLASSES,
          vectorLayers: state.vectorLayers,
          featureFlags: state.featureFlags,
          outputPath: state.outputPath || undefined,
          exportFormat: "img" as const,
          taskId,
          tileMode: state.performance.useTiling,
          tileMaxPixels: tileSizeToPixels(state.performance.tileSize, state.performance.suggestedTileSide),
          tileWorkers: state.performance.tileWorkers,
          detectShadows: state.classification.detectShadows,
          maxThreads: state.performance.useMaxThreads
            ? navigator.hardwareConcurrency ?? null
            : null,
        };
        const result = state.vectorLayers.length
          ? await runFullPipeline(params)
          : await runStep1(params);
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "MEA cancelled" });
        } else {
          handleSingleResult(result, "MEA");
        }
      } catch (e: any) {
        dispatch({ type: "SET_STATUS", text: `MEA error: ${e.message}` });
      } finally {
        activeTaskIdRef.current = null;
        stopProgress();
        dispatch({ type: "SET_PROGRESS", progress: null });
      }
    }
    dispatch({ type: "SET_RUNNING", step: "idle" });
  };

  const handleSingleResult = (result: any, label: string) => {
    const elapsed = formatElapsed(Date.now() - runStartRef.current);
    if (result.status === "ok") {
      const path = result.outputPath || result.saved?.[0] || "";
      const classifiedPath: string = result.classifiedPath || "";
      dispatch({ type: "SET_STATUS", text: `${label} complete ✓ (${elapsed})` });
      if (path) {
        dispatch({ type: "SET_LAST_RESULT_PATH", path });

        const tileOutputs: string[] | undefined = result.tileOutputs;
        const outputPaths: string[] = [];
        // When vectors were used, only show the with_vectors output (path),
        // not the classified output. When no vectors, path == classifiedPath.
        if (tileOutputs && tileOutputs.length > 0) {
          outputPaths.push(...tileOutputs);
        } else {
          outputPaths.push(path);
        }
        addResultsToGroup(label, outputPaths);
      }
    } else {
      dispatch({
        type: "SET_STATUS",
        text: `${label} failed: ${result.message ?? "Unknown error"}`,
      });
    }
  };

  return (
    <div className="pt-2 pb-1 px-1 border-t border-surface-700 mt-1 space-y-1.5">
      <div className="grid grid-cols-2 gap-1.5">
        <ActionBtn
          label="Step 1: Classify"
          onClick={handleStep1}
          disabled={isRunning}
          color="blue"
        />
        <ActionBtn
          label="Step 2: Vectors"
          onClick={handleStep2}
          disabled={isRunning}
          color="blue"
        />
        <ActionBtn
          label="Full Pipeline"
          onClick={handleFull}
          disabled={isRunning}
          color="indigo"
        />
      </div>
      <ActionBtn
        label="Run for MEA"
        onClick={handleMEA}
        disabled={isRunning}
        color="emerald"
        full
      />
      {isRunning && (
        <button
          className="w-full text-xs font-medium py-1.5 px-2 rounded transition-colors bg-red-600 hover:bg-red-700 text-white"
          onClick={handleCancel}
        >
          ✕ Cancel
        </button>
      )}
    </div>
  );
}

function ActionBtn({
  label,
  onClick,
  disabled,
  color,
  full,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  color: "blue" | "indigo" | "emerald";
  full?: boolean;
}) {
  const base =
    "text-xs font-medium py-1.5 px-2 rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const colorMap = {
    blue: "bg-primary-600 hover:bg-primary-700 text-white",
    indigo: "bg-indigo-600 hover:bg-indigo-700 text-white",
    emerald: "bg-emerald-600 hover:bg-emerald-700 text-white",
  };
  return (
    <button
      className={`${base} ${colorMap[color]} ${full ? "col-span-2" : ""}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
}

function tileSizeToPixels(size: string, suggestedSide?: number | null): number {
  if (size === "Auto") {
    const side = suggestedSide ?? 1024;
    return side * side;
  }
  const s = parseInt(size) || 1024;
  return s * s;
}

function looksLikeDirectoryPath(path: string): boolean {
  const normalized = (path || "").trim();
  if (!normalized) return true;
  if (/[\\/]$/.test(normalized)) return true;
  return !/\.[a-zA-Z0-9]{1,6}$/.test(normalized);
}
