import { useCallback, useRef, useState } from "react";
import { useAppState, useAppDispatch } from "../../store";
import {
  runStep1,
  runStep2,
  runFullPipeline,
  runBatchClassify,
  extractRoads,
  mergeRoadMask,
  extractFeatures,
  mergeFeatureMasks,
  generateTaskId,
  startProgressStream,
  cancelTask,
  getRoadExtractConfig,
  setSam3Path,
} from "../../api/client";
import type { BatchResult, RoadExtractConfig, ExtractFeaturesResult } from "../../api/client";
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
  const [roadMasks, setRoadMasks] = useState<ExtractFeaturesResult | null>(null);
  const [buildingMasks, setBuildingMasks] = useState<ExtractFeaturesResult | null>(null);
  const [treeMasks, setTreeMasks] = useState<ExtractFeaturesResult | null>(null);
  const [fieldsMasks, setFieldsMasks] = useState<ExtractFeaturesResult | null>(null);
  const [roadConfig, setRoadConfig] = useState<RoadExtractConfig | null>(null);
  const [sam3PathInput, setSam3PathInput] = useState("");
  // All classification output paths from the last run (one per input image / tile-dir).
  // Merges iterate over these so every classification file gets its features applied.
  const [classificationPaths, setClassificationPaths] = useState<string[]>([]);

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
        // Track all per-image classification paths so merges apply to every file.
        setClassificationPaths(paths);
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

  const handleExtractRoads = async () => {
    const rasterFiles = getRasterFiles();
    if (rasterFiles.length === 0) {
      if (!state.rasterPath) return alert("Select a raster image first.");
      rasterFiles.push(state.rasterPath);
    }
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step1" });
    dispatch({ type: "SET_STATUS", text: `Extracting roads from ${rasterFiles.length} file(s)…` });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    activeTaskIdRef.current = taskId;
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const allMaskPaths: string[] = [];
      const allColors: [number, number, number][] = [];
      let skippedCount = 0;
      for (const file of rasterFiles) {
        const fileName = file.split(/[\\/]/).pop() || file;
        dispatch({ type: "SET_STATUS", text: `Extracting roads: ${fileName}…` });
        const result = await extractRoads({
          rasterPath: file,
          outputPath: state.outputPath || undefined,
          taskId,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Road extraction cancelled" });
          return;
        }
        if ((result as any).status === "skipped") {
          skippedCount++;
          console.log(`Roads skipped for ${fileName}: ${(result as any).message}`);
          continue;
        }
        if (result.status === "ok" && result.outputPath) {
          allMaskPaths.push(result.outputPath);
          allColors.push([45, 45, 48]);
        } else {
          console.warn(`Roads failed for ${fileName}: ${(result as any).message}`);
        }
      }
      const elapsed = formatElapsed(Date.now() - runStartRef.current);
      const skippedMsg = skippedCount > 0 ? ` (${skippedCount} skipped — no roads detected)` : "";
      if (allMaskPaths.length > 0) {
        setRoadMasks({ status: "ok", maskPaths: allMaskPaths, colors: allColors });
        dispatch({ type: "SET_STATUS", text: `Roads extracted: ${allMaskPaths.length} mask(s) (${elapsed})${skippedMsg}` });
        addResultsToGroup("Road Masks (SAM)", allMaskPaths);
      } else {
        dispatch({ type: "SET_STATUS", text: `Road extraction: no roads found in any file${skippedMsg}` });
      }
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Road extraction error: ${e.message}` });
    } finally {
      activeTaskIdRef.current = null;
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleMergeRoadMask = async () => {
    handleMergeFeature(roadMasks, "Roads");
  };

  const handleExtractFeature = async (featureType: "buildings" | "trees" | "fields", label: string) => {
    const rasterFiles = getRasterFiles();
    if (rasterFiles.length === 0 && state.rasterPath) rasterFiles.push(state.rasterPath);
    if (rasterFiles.length === 0) return alert("Select a raster image first.");
    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step1" });
    dispatch({ type: "SET_STATUS", text: `Extracting ${label} from ${rasterFiles.length} file(s)…` });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    activeTaskIdRef.current = taskId;
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const allMaskPaths: string[] = [];
      const allColors: [number, number, number][] = [];
      let skippedCount = 0;
      for (const file of rasterFiles) {
        const fileName = file.split(/[\\/]/).pop() || file;
        dispatch({ type: "SET_STATUS", text: `Extracting ${label}: ${fileName}…` });
        const result = await extractFeatures({
          rasterPath: file,
          featureType,
          outputPath: state.outputPath || undefined,
          taskId,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: `${label} extraction cancelled` });
          return;
        }
        if ((result as any).status === "skipped") {
          skippedCount++;
          console.log(`${label} skipped for ${fileName}: ${result.message}`);
          continue;
        }
        if (result.status === "ok" && result.maskPaths?.length) {
          allMaskPaths.push(...result.maskPaths);
          allColors.push(...result.colors!);
        } else {
          console.warn(`${label} failed for ${fileName}: ${(result as any).message}`);
        }
      }
      const elapsed = formatElapsed(Date.now() - runStartRef.current);
      const skippedMsg = skippedCount > 0 ? ` (${skippedCount} skipped — no ${label.toLowerCase()} detected)` : "";
      if (allMaskPaths.length > 0) {
        const combined: ExtractFeaturesResult = { status: "ok", maskPaths: allMaskPaths, colors: allColors };
        if (featureType === "buildings") setBuildingMasks(combined);
        else if (featureType === "trees") setTreeMasks(combined);
        else setFieldsMasks(combined);
        dispatch({ type: "SET_STATUS", text: `${label} extracted: ${allMaskPaths.length} mask(s) (${elapsed})${skippedMsg}` });
        addResultsToGroup(`${label} Masks`, allMaskPaths);
      } else {
        dispatch({ type: "SET_STATUS", text: `${label}: no features found in any file${skippedMsg}` });
      }
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `${label} extraction error: ${e.message}` });
    } finally {
      activeTaskIdRef.current = null;
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleMergeFeature = async (
    masks: ExtractFeaturesResult | null,
    label: string,
  ) => {
    if (!masks?.maskPaths?.length) return alert(`Run 'Extract ${label}' first.`);
    // Use tracked classification paths; fall back to lastResultPath for legacy compat.
    const clsPaths = classificationPaths.length > 0
      ? classificationPaths
      : state.lastResultPath ? [state.lastResultPath] : [];
    if (!clsPaths.length) return alert("Run classification first to produce an output.");

    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step2" });
    dispatch({ type: "SET_STATUS", text: `Merging ${label} into ${clsPaths.length} classification file(s)…` });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    activeTaskIdRef.current = taskId;
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const newClsPaths: string[] = [];
      const allDisplayPaths: string[] = [];

      for (let i = 0; i < clsPaths.length; i++) {
        const clsPath = clsPaths[i];
        if (clsPaths.length > 1) {
          dispatch({ type: "SET_STATUS", text: `Merging ${label} (${i + 1}/${clsPaths.length})…` });
        }
        const result = await mergeFeatureMasks({
          classificationPath: clsPath,
          maskPaths: masks.maskPaths,
          colors: masks.colors!,
          outputPath: state.outputPath || undefined,
          taskId,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: `${label} merge cancelled` });
          return;
        }
        if (result.status === "ok" && result.outputPath) {
          // Keep the output path (file or dir) for chaining into the next merge
          newClsPaths.push(result.outputPath);
          const tileOutputs = (result as any).tileOutputs as string[] | undefined;
          allDisplayPaths.push(...(tileOutputs?.length ? tileOutputs : [result.outputPath]));
        } else {
          console.warn(`${label} merge failed for ${clsPath}: ${(result as any).message}`);
        }
      }

      const elapsed = formatElapsed(Date.now() - runStartRef.current);
      if (allDisplayPaths.length > 0) {
        // Update classification paths to the merged outputs for chaining
        setClassificationPaths(newClsPaths);
        dispatch({ type: "SET_LAST_RESULT_PATH", path: newClsPaths[newClsPaths.length - 1] });
        dispatch({ type: "SET_STATUS", text: `${label} merged (${elapsed})` });
        addResultsToGroup(`Classification + ${label}`, allDisplayPaths);
      } else {
        dispatch({ type: "SET_STATUS", text: `${label} merge failed for all files` });
      }
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `${label} merge error: ${e.message}` });
    } finally {
      activeTaskIdRef.current = null;
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleMergeAll = async () => {
    const clsPaths = classificationPaths.length > 0
      ? classificationPaths
      : state.lastResultPath ? [state.lastResultPath] : [];
    if (!clsPaths.length) return alert("Run classification first.");

    const allMaskPaths: string[] = [];
    const allColors: [number, number, number][] = [];
    if (roadMasks?.maskPaths) { allMaskPaths.push(...roadMasks.maskPaths); allColors.push(...roadMasks.colors!); }
    if (buildingMasks?.maskPaths) { allMaskPaths.push(...buildingMasks.maskPaths); allColors.push(...buildingMasks.colors!); }
    if (treeMasks?.maskPaths) { allMaskPaths.push(...treeMasks.maskPaths); allColors.push(...treeMasks.colors!); }
    if (fieldsMasks?.maskPaths) { allMaskPaths.push(...fieldsMasks.maskPaths); allColors.push(...fieldsMasks.colors!); }
    if (!allMaskPaths.length) return alert("Extract at least one feature first.");

    runStartRef.current = Date.now();
    dispatch({ type: "SET_RUNNING", step: "step2" });
    dispatch({ type: "SET_STATUS", text: `Merging all features into ${clsPaths.length} classification file(s)…` });
    dispatch({ type: "SET_PROGRESS", progress: null });
    const taskId = generateTaskId();
    activeTaskIdRef.current = taskId;
    const stopProgress = startProgressStream(taskId, (evt) => {
      dispatch({ type: "SET_PROGRESS", progress: evt });
    });
    try {
      const newClsPaths: string[] = [];
      const allDisplayPaths: string[] = [];

      for (let i = 0; i < clsPaths.length; i++) {
        const clsPath = clsPaths[i];
        if (clsPaths.length > 1) {
          dispatch({ type: "SET_STATUS", text: `Merging all features (${i + 1}/${clsPaths.length})…` });
        }
        const result = await mergeFeatureMasks({
          classificationPath: clsPath,
          maskPaths: allMaskPaths,
          colors: allColors,
          outputPath: state.outputPath || undefined,
          taskId,
        });
        if ((result as any).status === "cancelled") {
          dispatch({ type: "SET_STATUS", text: "Merge all cancelled" });
          return;
        }
        if (result.status === "ok" && result.outputPath) {
          newClsPaths.push(result.outputPath);
          const tileOutputs = (result as any).tileOutputs as string[] | undefined;
          allDisplayPaths.push(...(tileOutputs?.length ? tileOutputs : [result.outputPath]));
        } else {
          console.warn(`Merge all failed for ${clsPath}: ${(result as any).message}`);
        }
      }

      const elapsed = formatElapsed(Date.now() - runStartRef.current);
      if (allDisplayPaths.length > 0) {
        setClassificationPaths(newClsPaths);
        dispatch({ type: "SET_LAST_RESULT_PATH", path: newClsPaths[newClsPaths.length - 1] });
        dispatch({ type: "SET_STATUS", text: `All features merged (${elapsed})` });
        addResultsToGroup("Classification + All Features", allDisplayPaths);
      } else {
        dispatch({ type: "SET_STATUS", text: `Merge all failed for all files` });
      }
    } catch (e: any) {
      dispatch({ type: "SET_STATUS", text: `Merge all error: ${e.message}` });
    } finally {
      activeTaskIdRef.current = null;
      stopProgress();
      dispatch({ type: "SET_PROGRESS", progress: null });
      dispatch({ type: "SET_RUNNING", step: "idle" });
    }
  };

  const handleSingleResult = (result: any, label: string) => {
    const elapsed = formatElapsed(Date.now() - runStartRef.current);
    if (result.status === "ok") {
      const path = result.outputPath || result.saved?.[0] || "";
      dispatch({ type: "SET_STATUS", text: `${label} complete ✓ (${elapsed})` });
      if (path) {
        dispatch({ type: "SET_LAST_RESULT_PATH", path });

        const tileOutputs: string[] | undefined = result.tileOutputs;
        const outputPaths: string[] = [];
        if (tileOutputs && tileOutputs.length > 0) {
          outputPaths.push(...tileOutputs);
        } else {
          outputPaths.push(path);
        }
        // Track classification paths for subsequent merges.
        // Use `path` (directory for tile mode, file for single mode) so the
        // merge backend receives the right input for geographic alignment.
        setClassificationPaths([path]);
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
      <div className="pt-1 border-t border-surface-700 space-y-1.5">
        {/* Header row */}
        <div className="text-[10px] text-surface-400 font-medium uppercase tracking-wide px-0.5">
          SAM Feature Extraction
        </div>

        {/* Roads */}
        <div className="grid grid-cols-2 gap-1.5">
          <ActionBtn
            label="Extract Roads"
            onClick={handleExtractRoads}
            disabled={isRunning}
            color="amber"
          />
          <ActionBtn
            label="Merge Roads"
            onClick={handleMergeRoadMask}
            disabled={isRunning || !roadMasks?.maskPaths?.length || (classificationPaths.length === 0 && !state.lastResultPath)}
            color="amber"
          />
        </div>

        {/* Buildings */}
        <div className="grid grid-cols-2 gap-1.5">
          <ActionBtn
            label="Extract Buildings"
            onClick={() => handleExtractFeature("buildings", "Buildings")}
            disabled={isRunning}
            color="amber"
          />
          <ActionBtn
            label="Merge Buildings"
            onClick={() => handleMergeFeature(buildingMasks, "Buildings")}
            disabled={isRunning || !buildingMasks || (classificationPaths.length === 0 && !state.lastResultPath)}
            color="amber"
          />
        </div>

        {/* Trees */}
        <div className="grid grid-cols-2 gap-1.5">
          <ActionBtn
            label="Extract Trees"
            onClick={() => handleExtractFeature("trees", "Trees")}
            disabled={isRunning}
            color="amber"
          />
          <ActionBtn
            label="Merge Trees"
            onClick={() => handleMergeFeature(treeMasks, "Trees")}
            disabled={isRunning || !treeMasks || (classificationPaths.length === 0 && !state.lastResultPath)}
            color="amber"
          />
        </div>

        {/* Fields */}
        <div className="grid grid-cols-2 gap-1.5">
          <ActionBtn
            label="Extract Fields"
            onClick={() => handleExtractFeature("fields", "Fields")}
            disabled={isRunning}
            color="amber"
          />
          <ActionBtn
            label="Merge Fields"
            onClick={() => handleMergeFeature(fieldsMasks, "Fields")}
            disabled={isRunning || !fieldsMasks || (classificationPaths.length === 0 && !state.lastResultPath)}
            color="amber"
          />
        </div>

        {/* Merge All */}
        <ActionBtn
          label="Merge All Features"
          onClick={handleMergeAll}
          disabled={isRunning || (!roadMasks && !buildingMasks && !treeMasks && !fieldsMasks) || (classificationPaths.length === 0 && !state.lastResultPath)}
          color="orange"
          full
        />

        {/* Status indicators */}
        <div className="text-[9px] text-surface-500 flex gap-2 flex-wrap px-0.5">
          <span className={roadMasks ? "text-green-500" : ""}>
            {roadMasks ? `✓ Roads (${roadMasks.maskPaths?.length})` : "· Roads"}
          </span>
          <span className={buildingMasks ? "text-green-500" : ""}>
            {buildingMasks ? `✓ Buildings (${buildingMasks.maskPaths?.length})` : "· Buildings"}
          </span>
          <span className={treeMasks ? "text-green-500" : ""}>
            {treeMasks ? `✓ Trees (${treeMasks.maskPaths?.length})` : "· Trees"}
          </span>
          <span className={fieldsMasks ? "text-green-500" : ""}>
            {fieldsMasks ? `✓ Fields (${fieldsMasks.maskPaths?.length})` : "· Fields"}
          </span>
        </div>

        {/* SAM3 backend config */}
        <details
          className="text-[10px] text-surface-400"
          onToggle={async (e) => {
            if ((e.target as HTMLDetailsElement).open && !roadConfig) {
              try { setRoadConfig(await getRoadExtractConfig()); } catch {}
            }
          }}
        >
          <summary className="cursor-pointer hover:text-surface-200">
            SAM backend: {roadConfig?.loadedBackend ?? "click to check"}
          </summary>
          {roadConfig && (
            <div className="mt-1 space-y-1 text-surface-500">
              <div>Backend: <span className="text-surface-300">{roadConfig.loadedBackend ?? "not loaded yet"}</span></div>
              <div>SAM3 dir: <span className="text-surface-300">{roadConfig.sam3LocalDir ?? "not found"}</span></div>
              <div>SAM3 checkpoint: <span className={roadConfig.sam3CheckpointFound ? "text-green-400" : "text-red-400"}>
                {roadConfig.sam3CheckpointFound ? "✓ found" : "✗ missing"}
              </span></div>
              <div className="flex gap-1 items-center">
                <input
                  className="flex-1 bg-surface-800 text-surface-200 text-[10px] px-1 py-0.5 rounded border border-surface-600"
                  placeholder="Path to sam3 folder…"
                  value={sam3PathInput}
                  onChange={(e) => setSam3PathInput(e.target.value)}
                />
                <button
                  className="px-1.5 py-0.5 bg-amber-700 hover:bg-amber-600 text-white rounded text-[10px]"
                  onClick={async () => {
                    const cfg = await setSam3Path(sam3PathInput || null);
                    setRoadConfig(cfg);
                  }}
                >Set</button>
              </div>
            </div>
          )}
        </details>
      </div>
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
  color: "blue" | "indigo" | "emerald" | "amber" | "orange";
  full?: boolean;
}) {
  const base =
    "text-xs font-medium py-1.5 px-2 rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const colorMap = {
    blue: "bg-primary-600 hover:bg-primary-700 text-white",
    indigo: "bg-indigo-600 hover:bg-indigo-700 text-white",
    emerald: "bg-emerald-600 hover:bg-emerald-700 text-white",
    amber: "bg-amber-600 hover:bg-amber-700 text-white",
    orange: "bg-orange-600 hover:bg-orange-700 text-white",
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
