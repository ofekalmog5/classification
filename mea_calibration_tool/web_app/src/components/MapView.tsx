import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useAppState } from "../store";
import CalibrationOverlay from "./CalibrationOverlay";
import { getRasterInfo, getRasterAsPng } from "../api/client";

interface RasterMeta { width: number; height: number; }

export default function MapView() {
  const { rasterPath, calibration } = useAppState();
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<L.ImageOverlay | null>(null);
  const [rasterMeta, setRasterMeta] = useState<RasterMeta | null>(null);

  // Initialize Leaflet map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    mapRef.current = L.map(containerRef.current, {
      crs: L.CRS.Simple,
      zoom: 0,
      minZoom: -4,
      maxZoom: 8,
      zoomControl: true,
      attributionControl: false,
    });
  }, []);

  // Load raster when path changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !rasterPath) return;

    let cancelled = false;

    (async () => {
      try {
        const [info, pngRes] = await Promise.all([
          getRasterInfo(rasterPath),
          getRasterAsPng(rasterPath),
        ]);

        if (cancelled) return;

        overlayRef.current?.remove();

        const dataUrl = `data:image/png;base64,${pngRes.image_base64}`;

        // Use pixel-space bounds so containerPointToLatLng returns pixel coordinates.
        // CRS.Simple: lat = y (increases up), lng = x (increases right).
        // We place image from SW=[0,0] to NE=[height,width], so:
        //   raster pixel (px, py)  →  lat = height - py,  lng = px
        const bounds: L.LatLngBoundsLiteral = [[0, 0], [info.height, info.width]];
        overlayRef.current = L.imageOverlay(dataUrl, bounds).addTo(map);
        map.fitBounds(bounds);
        setRasterMeta({ width: info.width, height: info.height });
      } catch (e) {
        console.error("[MapView] load error:", e);
      }
    })();

    return () => { cancelled = true; };
  }, [rasterPath]);

  return (
    <div style={{ position: "relative", flex: 1, background: "#111" }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      {calibration.active && mapRef.current && rasterMeta && (
        <CalibrationOverlay
          map={mapRef.current}
          rasterWidth={rasterMeta.width}
          rasterHeight={rasterMeta.height}
        />
      )}
    </div>
  );
}
