import React, { createContext, useContext, useReducer, type Dispatch } from "react";
import type {
  AppState,
  ClassItem,
  VectorLayer,
  MapLayer,
  LayerGroup,
  ProgressEvent,
  PipelineStep,
  FeatureFlags,
  PerformanceSettings,
  ClassificationSettings,
  ImageryMode,
} from "../types";

/* ── Initial state ───────────────────────────────────────────────── */

const cpuCount = navigator.hardwareConcurrency ?? 4;

export const initialState: AppState = {
  rasterPath: "",
  outputPath: "",
  lastResultPath: "",
  imageryMode: "regular",
  featureFlags: { spectral: true, texture: true, indices: false, colorIndices: true, entropy: false, morphCleanup: true },
  classes: [],
  classCount: 3,
  vectorLayers: [],
  performance: {
    useTiling: false,
    tileSize: "Auto",
    tileWorkers: cpuCount,
    imageWorkers: Math.min(4, cpuCount),
    useMaxThreads: false,
  },
  classification: {
    detectShadows: false,
    shareModel: true,
    exportFormat: "tif",
  },
  mapLayers: [],
  layerGroups: [],
  running: "idle",
  progress: null,
  statusText: "Ready",
};

/* ── Action types ────────────────────────────────────────────────── */

export type Action =
  | { type: "SET_RASTER_PATH"; path: string }
  | { type: "SET_OUTPUT_PATH"; path: string }
  | { type: "SET_LAST_RESULT_PATH"; path: string }
  | { type: "SET_IMAGERY_MODE"; mode: ImageryMode }
  | { type: "SET_FEATURE_FLAGS"; flags: Partial<FeatureFlags> }
  | { type: "SET_CLASS_COUNT"; count: number }
  | { type: "SET_CLASSES"; classes: ClassItem[] }
  | { type: "ADD_VECTOR"; layer: VectorLayer }
  | { type: "REMOVE_VECTOR"; id: string }
  | { type: "MOVE_VECTOR"; id: string; direction: -1 | 1 }
  | { type: "SET_VECTORS"; layers: VectorLayer[] }
  | { type: "SET_PERFORMANCE"; settings: Partial<PerformanceSettings> }
  | { type: "SET_CLASSIFICATION"; settings: Partial<ClassificationSettings> }
  // Map
  | { type: "ADD_MAP_LAYER"; layer: MapLayer }
  | { type: "REMOVE_MAP_LAYER"; id: string }
  | { type: "TOGGLE_MAP_LAYER"; id: string }
  | { type: "SET_LAYER_OPACITY"; id: string; opacity: number }
  | { type: "ADD_LAYER_GROUP"; group: LayerGroup }
  | { type: "REMOVE_LAYER_GROUP"; id: string }
  | { type: "TOGGLE_LAYER_GROUP"; id: string }
  | { type: "ADD_TO_GROUP"; layerId: string; groupId: string }
  | { type: "ADD_MANY_TO_GROUP"; layerIds: string[]; groupId: string }
  | { type: "REMOVE_FROM_GROUP"; layerId: string; groupId: string }
  | { type: "SET_ALL_LAYERS_VISIBLE"; visible: boolean }
  | { type: "SET_LAYERS_VISIBLE"; ids: string[]; visible: boolean }
  | { type: "REMOVE_ALL_MAP_LAYERS" }
  // Run state
  | { type: "SET_RUNNING"; step: PipelineStep }
  | { type: "SET_PROGRESS"; progress: ProgressEvent | null }
  | { type: "SET_STATUS"; text: string };

/* ── Reducer ─────────────────────────────────────────────────────── */

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_RASTER_PATH":
      return { ...state, rasterPath: action.path };

    case "SET_OUTPUT_PATH":
      return { ...state, outputPath: action.path };

    case "SET_LAST_RESULT_PATH":
      return { ...state, lastResultPath: action.path };

    case "SET_IMAGERY_MODE": {
      const flags =
        action.mode === "regular"
          ? { ...state.featureFlags, indices: false, colorIndices: true }
          : state.featureFlags;
      return { ...state, imageryMode: action.mode, featureFlags: flags };
    }

    case "SET_FEATURE_FLAGS":
      return {
        ...state,
        featureFlags: { ...state.featureFlags, ...action.flags },
      };

    case "SET_CLASS_COUNT":
      return { ...state, classCount: Math.max(2, action.count) };

    case "SET_CLASSES":
      return { ...state, classes: action.classes };

    case "ADD_VECTOR":
      return {
        ...state,
        vectorLayers: [...state.vectorLayers, action.layer],
      };

    case "REMOVE_VECTOR":
      return {
        ...state,
        vectorLayers: state.vectorLayers.filter((l) => l.id !== action.id),
      };

    case "MOVE_VECTOR": {
      const idx = state.vectorLayers.findIndex((l) => l.id === action.id);
      if (idx < 0) return state;
      const newIdx = idx + action.direction;
      if (newIdx < 0 || newIdx >= state.vectorLayers.length) return state;
      const arr = [...state.vectorLayers];
      [arr[idx], arr[newIdx]] = [arr[newIdx], arr[idx]];
      return { ...state, vectorLayers: arr };
    }

    case "SET_VECTORS":
      return { ...state, vectorLayers: action.layers };

    case "SET_PERFORMANCE":
      return {
        ...state,
        performance: { ...state.performance, ...action.settings },
      };

    case "SET_CLASSIFICATION":
      return {
        ...state,
        classification: { ...state.classification, ...action.settings },
      };

    /* ── Map layers ────────────────────────────────────────────────── */

    case "ADD_MAP_LAYER":
      return { ...state, mapLayers: [...state.mapLayers, action.layer] };

    case "REMOVE_MAP_LAYER":
      return {
        ...state,
        mapLayers: state.mapLayers.filter((l) => l.id !== action.id),
        layerGroups: state.layerGroups.map((g) => ({
          ...g,
          layerIds: g.layerIds.filter((lid) => lid !== action.id),
        })),
      };

    case "TOGGLE_MAP_LAYER":
      return {
        ...state,
        mapLayers: state.mapLayers.map((l) =>
          l.id === action.id ? { ...l, visible: !l.visible } : l
        ),
      };

    case "SET_LAYER_OPACITY":
      return {
        ...state,
        mapLayers: state.mapLayers.map((l) =>
          l.id === action.id ? { ...l, opacity: action.opacity } : l
        ),
      };

    case "ADD_LAYER_GROUP":
      return { ...state, layerGroups: [...state.layerGroups, action.group] };

    case "REMOVE_LAYER_GROUP":
      return {
        ...state,
        layerGroups: state.layerGroups.filter((g) => g.id !== action.id),
      };

    case "TOGGLE_LAYER_GROUP": {
      const grp = state.layerGroups.find((g) => g.id === action.id);
      if (!grp) return state;
      const newVis = !grp.visible;
      return {
        ...state,
        layerGroups: state.layerGroups.map((g) =>
          g.id === action.id ? { ...g, visible: newVis } : g
        ),
        mapLayers: state.mapLayers.map((l) =>
          grp.layerIds.includes(l.id) ? { ...l, visible: newVis } : l
        ),
      };
    }

    case "ADD_TO_GROUP":
      return {
        ...state,
        layerGroups: state.layerGroups.map((g) =>
          g.id === action.groupId && !g.layerIds.includes(action.layerId)
            ? { ...g, layerIds: [...g.layerIds, action.layerId] }
            : g
        ),
      };

    case "ADD_MANY_TO_GROUP": {
      const newIds = action.layerIds;
      return {
        ...state,
        // first remove these layers from any other group
        layerGroups: state.layerGroups.map((g) => {
          if (g.id === action.groupId) {
            const merged = [...g.layerIds, ...newIds.filter((id) => !g.layerIds.includes(id))];
            return { ...g, layerIds: merged };
          }
          // remove from other groups
          return { ...g, layerIds: g.layerIds.filter((id) => !newIds.includes(id)) };
        }),
      };
    }

    case "REMOVE_ALL_MAP_LAYERS":
      return {
        ...state,
        mapLayers: [],
        layerGroups: state.layerGroups.map((g) => ({ ...g, layerIds: [] })),
      };

    case "SET_ALL_LAYERS_VISIBLE":
      return {
        ...state,
        mapLayers: state.mapLayers.map((l) => ({ ...l, visible: action.visible })),
        layerGroups: state.layerGroups.map((g) => ({ ...g, visible: action.visible })),
      };

    case "SET_LAYERS_VISIBLE":
      return {
        ...state,
        mapLayers: state.mapLayers.map((l) =>
          action.ids.includes(l.id) ? { ...l, visible: action.visible } : l
        ),
      };

    case "REMOVE_FROM_GROUP":
      return {
        ...state,
        layerGroups: state.layerGroups.map((g) =>
          g.id === action.groupId
            ? { ...g, layerIds: g.layerIds.filter((id) => id !== action.layerId) }
            : g
        ),
      };

    /* ── Run state ─────────────────────────────────────────────────── */

    case "SET_RUNNING":
      return { ...state, running: action.step };

    case "SET_PROGRESS":
      return { ...state, progress: action.progress };

    case "SET_STATUS":
      return { ...state, statusText: action.text };

    default:
      return state;
  }
}

/* ── Context ─────────────────────────────────────────────────────── */

const StateCtx = createContext<AppState>(initialState);
const DispatchCtx = createContext<Dispatch<Action>>(() => {});

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <StateCtx.Provider value={state}>
      <DispatchCtx.Provider value={dispatch}>{children}</DispatchCtx.Provider>
    </StateCtx.Provider>
  );
}

export function useAppState() {
  return useContext(StateCtx);
}

export function useAppDispatch() {
  return useContext(DispatchCtx);
}
