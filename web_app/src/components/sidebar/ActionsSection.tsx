import { useCallback } from "react";
import { useAppState, useAppDispatch } from "../../store";
import {
  runStep1,
  runStep2,
  runFullPipeline,
  runStep3,
  generateTaskId,
  startProgressStream,
} from "../../api/client";
import { MEA_CLASSES } from "../../constants/mea";

export default function ActionsSection() {
  const state = useAppState();
  const dispatch = useAppDispatch();
  const isRunning = state.running !== "idle";

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
      tileMaxPixels: tileSizeToPixels(state.performance.tileSize),
      tileWorkers: state.performance.tileWorkers,
      detectShadows: state.classification.detectShadows,
      maxThreads,
    };
  }, [state]);

  const handleStep1 = async () => {
    if (!state.rasterPath) return alert("Select a raster image first.");
    if (looksLikeDirectoryPath(state.rasterPath)) {
      return alert("Input raster must be a file, not a folder. Choose a raster file (.tif/.img/.jpg...).");
    }
    if (!state.classes.length) return alert("Define materials first.");
    dispatch({ type: "SET_RUNNING", step: "step1" });
    dispatch({ type: "SET_STATUS", text: "Running Step 1: Classification…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const result = await runStep1({
        rasterPath: state.rasterPath,
        taskId,
        ...commonParams(),
      });
      handleResult(result, "Step 1");
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Step 1 error: ${e.message}` });
    } finally {
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleStep2 = async () => {
    if (!state.lastResultPath) return alert("Run Step 1 first to produce a classification file.");
    if (!state.vectorLayers.length) return alert("Add at least one vector layer.");
    dispatch({ type: "SET_RUNNING", step: "step2" });
    dispatch({ type: "SET_STATUS", text: "Running Step 2: Vector rasterization…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
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
        tileMaxPixels: tileSizeToPixels(state.performance.tileSize),
        tileWorkers: state.performance.tileWorkers,
        maxThreads: state.performance.useMaxThreads
          ? navigator.hardwareConcurrency ?? null
          : null,
      });
      handleResult(result, "Step 2");
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Step 2 error: ${e.message}` });
    } finally {
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleStep3 = async () => {
    if (!state.lastResultPath) return alert("Run Step 1/2 first to produce a classification file.");
    dispatch({ type: "SET_RUNNING", step: "step3" });
    dispatch({ type: "SET_STATUS", text: "Running Step 3: Remove road objects…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const result = await runStep3({
        classificationPath: state.lastResultPath,
        taskId,
      });
      handleResult(result, "Step 3");
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Step 3 error: ${e.message}` });
    } finally {
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleFull = async () => {
    if (!state.rasterPath) return alert("Select a raster image first.");
    if (looksLikeDirectoryPath(state.rasterPath)) {
      return alert("Input raster must be a file, not a folder. Choose a raster file (.tif/.img/.jpg...).");
    }
    if (!state.classes.length) return alert("Define materials first.");
    dispatch({ type: "SET_RUNNING", step: "full" });
    dispatch({ type: "SET_STATUS", text: "Running Full Pipeline…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const result = await runFullPipeline({
        rasterPath: state.rasterPath,
        vectorLayers: state.vectorLayers,
        taskId,
        ...commonParams(),
      });
      handleResult(result, "Full Pipeline");
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Full pipeline error: ${e.message}` });
    } finally {
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleMEA = async () => {
    if (!state.rasterPath) return alert("Select a raster image first.");
    if (looksLikeDirectoryPath(state.rasterPath)) {
      return alert("MEA requires a single raster file. You selected a folder path.");
    }
    dispatch({ type: "SET_CLASSES", classes: MEA_CLASSES });
    dispatch({ type: "SET_CLASS_COUNT", count: MEA_CLASSES.length });
    // MEA default: export as .img
    dispatch({ type: "SET_CLASSIFICATION", settings: { exportFormat: "img" } });
    dispatch({ type: "SET_RUNNING", step: "mea" });
    dispatch({ type: "SET_STATUS", text: "Running MEA Classification…" });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const params = {
        rasterPath: state.rasterPath,
        classes: MEA_CLASSES,
        vectorLayers: state.vectorLayers,
        featureFlags: state.featureFlags,
        outputPath: state.outputPath || undefined,
        exportFormat: "img" as const,
        taskId,
        tileMode: state.performance.useTiling,
        tileMaxPixels: tileSizeToPixels(state.performance.tileSize),
        tileWorkers: state.performance.tileWorkers,
        detectShadows: state.classification.detectShadows,
        maxThreads: state.performance.useMaxThreads
          ? navigator.hardwareConcurrency ?? null
          : null,
      };
      const result = state.vectorLayers.length
        ? await runFullPipeline(params)
        : await runStep1(params);
      handleResult(result, "MEA");
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `MEA error: ${e.message}` });
    } finally {
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleResult = (result: any, label: string) => {
    if (result.status === "ok") {
      const path = result.outputPath || result.saved?.[0] || "";
      dispatch({ type: "SET_STATUS", text: `${label} complete ✓` });
      if (path) {
        // Track where the result was saved (for Step 2/3 input)
        dispatch({ type: "SET_LAST_RESULT_PATH", path });
        // Add result as a map layer
        dispatch({
          type: "ADD_MAP_LAYER",
          layer: {
            id: `result-${Date.now()}`,
            name: `${label} Result`,
            type: "classification-result",
            filePath: path,
            visible: true,
            opacity: 0.85,
          },
        });
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
          label="Step 3: Road Clean"
          onClick={handleStep3}
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

function tileSizeToPixels(size: string): number {
  if (size === "Auto") return 2048 * 2048;
  const s = parseInt(size) || 1024;
  return s * s;
}

function looksLikeDirectoryPath(path: string): boolean {
  const normalized = (path || "").trim();
  if (!normalized) return true;
  if (/[\\/]$/.test(normalized)) return true;
  return !/\.[a-zA-Z0-9]{1,6}$/.test(normalized);
}
