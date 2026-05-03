export interface MeaClass {
  id: string;
  name: string;
}

export interface CalibrationMaterialOverride {
  reference_color: string;
  reference_rgb: [number, number, number];
  anchors: [number, number, number][];
  tolerance_radius?: number;
  sample_count: number;
  sample_std_rgb: [number, number, number];
}

export interface CalibrationProfile {
  version: number;
  name: string;
  created_at: string;
  raster_path: string;
  material_overrides: Record<string, CalibrationMaterialOverride>;
  bias_overrides?: Record<string, number>;
  frequency_prior_overrides?: Record<string, number>;
}

export interface CalibrationSwatch {
  material: string;
  regions: Array<{ x: number; y: number; width: number; height: number }>;
  computedMeanRgb: [number, number, number] | null;
  computedStdRgb: [number, number, number] | null;
  sampleCount: number;
}

export interface CalibrationState {
  active: boolean;
  rasterPath: string;
  profileName: string;
  swatches: Record<string, CalibrationSwatch>;
  activeMaterial: string | null;
  savedProfile: CalibrationProfile | null;
  profilePath: string | null;
}

export interface AppState {
  rasterPath: string;
  calibration: CalibrationState;
  profileColors: Record<string, string>;
}
