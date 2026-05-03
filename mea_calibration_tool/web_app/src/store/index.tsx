import React, { createContext, useContext, useReducer, type Dispatch } from "react";
import type { AppState, CalibrationProfile, CalibrationSwatch } from "../types";
import { MEA_CLASSES, FACTORY_COLORS } from "../constants/mea";

const initialSwatches = () =>
  Object.fromEntries(
    MEA_CLASSES.map((c) => [
      c.name,
      { material: c.name, regions: [], computedMeanRgb: null, computedStdRgb: null, sampleCount: 0 } as CalibrationSwatch,
    ])
  );

export const initialState: AppState = {
  rasterPath: "",
  calibration: {
    active: false,
    rasterPath: "",
    profileName: "",
    swatches: {},
    activeMaterial: null,
    savedProfile: null,
    profilePath: null,
  },
  profileColors: { ...FACTORY_COLORS },
};

export type Action =
  | { type: "SET_RASTER_PATH"; path: string }
  | { type: "SET_PROFILE_COLORS"; colors: Record<string, string> }
  | { type: "CAL_OPEN"; rasterPath: string }
  | { type: "CAL_CLOSE" }
  | { type: "CAL_SET_ACTIVE_MATERIAL"; material: string | null }
  | { type: "CAL_ADD_REGION"; material: string; region: { x: number; y: number; width: number; height: number } }
  | { type: "CAL_CLEAR_REGIONS"; material: string }
  | { type: "CAL_SET_SWATCH_RESULT"; material: string; meanRgb: [number, number, number]; stdRgb: [number, number, number]; sampleCount: number }
  | { type: "CAL_SET_PROFILE"; profile: CalibrationProfile | null; profilePath: string | null }
  | { type: "CAL_SET_NAME"; name: string };

function reducer(state: AppState, action: Action): AppState {
  const cal = state.calibration;
  switch (action.type) {
    case "SET_RASTER_PATH":
      return { ...state, rasterPath: action.path };

    case "SET_PROFILE_COLORS":
      return { ...state, profileColors: action.colors };

    case "CAL_OPEN":
      return {
        ...state,
        calibration: {
          ...cal,
          active: true,
          rasterPath: action.rasterPath,
          swatches: initialSwatches(),
          activeMaterial: null,
        },
      };

    case "CAL_CLOSE":
      return { ...state, calibration: { ...cal, active: false, activeMaterial: null } };

    case "CAL_SET_ACTIVE_MATERIAL":
      return { ...state, calibration: { ...cal, activeMaterial: action.material } };

    case "CAL_ADD_REGION": {
      const prev = cal.swatches[action.material] ?? {
        material: action.material, regions: [], computedMeanRgb: null, computedStdRgb: null, sampleCount: 0,
      };
      return {
        ...state,
        calibration: {
          ...cal,
          swatches: {
            ...cal.swatches,
            [action.material]: { ...prev, regions: [...prev.regions, action.region] },
          },
        },
      };
    }

    case "CAL_CLEAR_REGIONS": {
      const prev = cal.swatches[action.material];
      if (!prev) return state;
      return {
        ...state,
        calibration: {
          ...cal,
          swatches: {
            ...cal.swatches,
            [action.material]: { ...prev, regions: [], computedMeanRgb: null, computedStdRgb: null, sampleCount: 0 },
          },
        },
      };
    }

    case "CAL_SET_SWATCH_RESULT": {
      const prev = cal.swatches[action.material] ?? {
        material: action.material, regions: [], computedMeanRgb: null, computedStdRgb: null, sampleCount: 0,
      };
      return {
        ...state,
        calibration: {
          ...cal,
          swatches: {
            ...cal.swatches,
            [action.material]: {
              ...prev,
              computedMeanRgb: action.meanRgb,
              computedStdRgb: action.stdRgb,
              sampleCount: action.sampleCount,
            },
          },
        },
      };
    }

    case "CAL_SET_PROFILE": {
      const colors: Record<string, string> = { ...FACTORY_COLORS };
      if (action.profile) {
        for (const [name, mo] of Object.entries(action.profile.material_overrides)) {
          if (mo.reference_color) colors[name] = mo.reference_color;
        }
      }
      return {
        ...state,
        profileColors: colors,
        calibration: {
          ...cal,
          savedProfile: action.profile,
          profilePath: action.profilePath,
        },
      };
    }

    case "CAL_SET_NAME":
      return { ...state, calibration: { ...cal, profileName: action.name } };

    default:
      return state;
  }
}

const StateCtx = createContext<AppState>(initialState);
const DispatchCtx = createContext<Dispatch<Action>>(() => {});

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <StateCtx.Provider value={state}>
      <DispatchCtx.Provider value={dispatch}>{children}</DispatchCtx.Provider>
    </StateCtx.Provider>
  );
}

export const useAppState = () => useContext(StateCtx);
export const useAppDispatch = () => useContext(DispatchCtx);
