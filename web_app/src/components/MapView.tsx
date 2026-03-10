import { useEffect, useRef, useCallback, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useAppState } from "../store";

/* ─── Fetch WGS-84 bounds from backend ───────────────────────────── */
async function fetchBounds(
  filePath: string
): Promise<L.LatLngBoundsLiteral | null> {
  try {
    const r = await fetch("/api/raster-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filePath }),
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      console.error(`[MapView] raster-info failed (${r.status}):`, txt);
      return null;
    }
    const data = await r.json();
    if (data.bounds) return data.bounds as L.LatLngBoundsLiteral;
    console.error("[MapView] raster-info returned no bounds:", data);
  } catch (e) {
    console.error("[MapView] raster-info exception:", e);
  }
  return null;
}

/* ─── Fetch raster image as blob via POST (robust Windows paths) ── */
async function fetchRasterBlob(
  filePath: string,
  maxDim = 1536
): Promise<string | null> {
  try {
    const r = await fetch("/api/raster-as-png", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filePath, maxDim }),
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      console.error(`[MapView] raster-as-png failed (${r.status}):`, txt);
      return null;
    }
    const blob = await r.blob();
    return URL.createObjectURL(blob);
  } catch (e) {
    console.error("[MapView] raster-as-png exception:", e);
  }
  return null;
}

/**
 * Local-only Leaflet map.
 * Each raster is displayed as an L.imageOverlay:
 *   - backend POST /raster-as-png → converts any format (tif, img, …) to PNG/JPEG
 *   - backend POST /raster-info   → returns WGS-84 bounds
 * This guarantees correct geographic positioning regardless of source CRS.
 */
export default function MapView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<Record<string, L.ImageOverlay>>({});
  const boundsRef = useRef<Record<string, L.LatLngBounds>>({});
  const blobUrlsRef = useRef<Record<string, string>>({});
  const loadingRef = useRef<Set<string>>(new Set());
  const { mapLayers } = useAppState();
  const [queuePulse, setQueuePulse] = useState(0);
  const MAX_PARALLEL_LOADS = 4;

  // Initialise map (no base tile layer — local only)
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: [0, 0],
      zoom: 2,
      zoomControl: true,
      attributionControl: false,
      preferCanvas: true,
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Cleanup blob URLs when component unmounts
  useEffect(() => {
    return () => {
      for (const url of Object.values(blobUrlsRef.current)) {
        URL.revokeObjectURL(url);
      }
    };
  }, []);

  // Fit view to all visible layers
  const fitView = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    let combined: L.LatLngBounds | null = null;
    for (const ml of mapLayers) {
      if (!ml.visible) continue;
      const b = boundsRef.current[ml.id];
      if (b && b.isValid()) {
        combined = combined
          ? combined.extend(b)
          : L.latLngBounds(b.getSouthWest(), b.getNorthEast());
      }
    }
    if (combined && combined.isValid()) {
      map.fitBounds(combined, { padding: [20, 20], maxZoom: 22 });
    }
  }, [mapLayers]);

  // Sync map layers
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const currentIds = new Set(mapLayers.map((l) => l.id));

    // Remove layers no longer in state
    for (const id of Object.keys(layersRef.current)) {
      if (!currentIds.has(id)) {
        map.removeLayer(layersRef.current[id]);
        delete layersRef.current[id];
        delete boundsRef.current[id];
        // Revoke blob URL
        if (blobUrlsRef.current[id]) {
          URL.revokeObjectURL(blobUrlsRef.current[id]);
          delete blobUrlsRef.current[id];
        }
      }
    }

    // Add / update layers
    for (const ml of mapLayers) {
      const existing = layersRef.current[ml.id];

      if (existing) {
        // Toggle visibility
        if (ml.visible && !map.hasLayer(existing)) map.addLayer(existing);
        else if (!ml.visible && map.hasLayer(existing)) map.removeLayer(existing);
        // Update opacity
        existing.setOpacity(ml.opacity);
        continue;
      }

      // Skip if already loading
      if (loadingRef.current.has(ml.id)) continue;

      // Throttle concurrent raster requests to keep UI responsive
      if (loadingRef.current.size >= MAX_PARALLEL_LOADS) continue;

      if (ml.type === "raster-input" || ml.type === "classification-result") {
        // Only materialize raster overlays when they are visible
        if (!ml.visible) continue;

        loadingRef.current.add(ml.id);

        // 1. Fetch bounds + image in parallel
        Promise.all([
          fetchBounds(ml.filePath),
          fetchRasterBlob(ml.filePath),
        ])
          .then(([rawBounds, blobUrl]) => {
            if (!rawBounds) throw new Error(`No bounds for ${ml.filePath}`);
            if (!blobUrl) throw new Error(`No image for ${ml.filePath}`);

            const lb = L.latLngBounds(rawBounds);
            boundsRef.current[ml.id] = lb;
            blobUrlsRef.current[ml.id] = blobUrl;

            // 2. Create image overlay from blob URL
            const overlay = L.imageOverlay(blobUrl, lb, {
              opacity: ml.opacity,
              interactive: false,
            });

            layersRef.current[ml.id] = overlay;
            if (ml.visible) overlay.addTo(map);

            // Auto-fit on first layer
            if (Object.keys(layersRef.current).length === 1) {
              map.fitBounds(lb, { padding: [20, 20], maxZoom: 22 });
            }
          })
          .catch((err) => {
            console.warn(`[MapView] Failed to load layer "${ml.name}" (${ml.filePath}):`, err);
          })
          .finally(() => {
            loadingRef.current.delete(ml.id);
            // Trigger another sync pass so queued layers continue loading
            setQueuePulse((c) => c + 1);
          });
      } else if (ml.type === "vector-overlay") {
        // placeholder for vector layers
        const marker = L.marker([31.5, 34.5]).bindPopup(ml.name) as any;
        if (ml.visible) marker.addTo(map);
        layersRef.current[ml.id] = marker;
      }
    }
  }, [mapLayers, queuePulse]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />

      {/* Fit View button */}
      <button
        onClick={fitView}
        className="absolute top-3 right-3 z-[1000] bg-surface-800 hover:bg-surface-700 text-surface-200 text-xs font-medium px-3 py-1.5 rounded shadow-lg border border-surface-600 transition-colors"
        title="Fit view to all visible layers"
      >
        ⊞ Fit View
      </button>

      {/* Loading indicator */}
      {loadingRef.current.size > 0 && (
        <div className="absolute bottom-3 left-3 z-[1000] bg-surface-900/80 text-surface-300 text-xs px-2 py-1 rounded">
          Loading rasters…
        </div>
      )}
    </div>
  );
}
