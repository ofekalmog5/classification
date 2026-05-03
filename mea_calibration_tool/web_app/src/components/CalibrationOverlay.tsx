import { useRef, useCallback, useEffect, useState } from "react";
import L from "leaflet";
import { useAppState, useAppDispatch } from "../store";

interface Props {
  map: L.Map;
  rasterWidth: number;
  rasterHeight: number;
}

export default function CalibrationOverlay({ map, rasterWidth, rasterHeight }: Props) {
  const { calibration, profileColors } = useAppState();
  const dispatch = useAppDispatch();

  const overlayRef = useRef<HTMLDivElement>(null);

  // Polygon being drawn (screen coordinates relative to overlay)
  const [draftPts, setDraftPts] = useState<[number, number][]>([]);
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);

  // Force SVG re-render when map pans/zooms
  const [viewKey, setViewKey] = useState(0);
  useEffect(() => {
    const refresh = () => setViewKey((k) => k + 1);
    map.on("move zoom", refresh);
    return () => { map.off("move zoom", refresh); };
  }, [map]);

  const matColor = (name: string) => profileColors[name] ?? "#aaaaaa";

  // Screen offset (relative to overlay) → raster pixel
  const toRaster = useCallback(
    (sx: number, sy: number) => {
      const ll = map.containerPointToLatLng(L.point(sx, sy));
      return {
        x: Math.max(0, Math.min(rasterWidth - 1, Math.round(ll.lng))),
        y: Math.max(0, Math.min(rasterHeight - 1, Math.round(rasterHeight - ll.lat))),
      };
    },
    [map, rasterWidth, rasterHeight]
  );

  // Raster pixel → screen offset
  const toScreen = useCallback(
    (rx: number, ry: number): [number, number] => {
      const pt = map.latLngToContainerPoint(L.latLng(rasterHeight - ry, rx));
      return [pt.x, pt.y];
    },
    [map, rasterHeight]
  );

  // Click: add vertex to polygon
  const onClick = useCallback(
    (e: MouseEvent) => {
      if (!calibration.activeMaterial) return;
      // Ignore double-click's second click
      if (e.detail >= 2) return;
      e.preventDefault();
      const rect = overlayRef.current!.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      setDraftPts((pts) => [...pts, [sx, sy]]);
    },
    [calibration.activeMaterial]
  );

  // Double-click: close polygon, compute bounding box, dispatch
  const onDblClick = useCallback(
    (e: MouseEvent) => {
      if (!calibration.activeMaterial) return;
      e.preventDefault();
      e.stopPropagation();
      setDraftPts((pts) => {
        const allPts = pts.length >= 2 ? pts : pts; // need at least 2 distinct screen pts
        if (allPts.length < 2) return [];
        const rasterPts = allPts.map(([sx, sy]) => toRaster(sx, sy));
        const xs = rasterPts.map((p) => p.x);
        const ys = rasterPts.map((p) => p.y);
        const x0 = Math.min(...xs);
        const y0 = Math.min(...ys);
        const w = Math.max(1, Math.max(...xs) - x0);
        const h = Math.max(1, Math.max(...ys) - y0);
        dispatch({
          type: "CAL_ADD_REGION",
          material: calibration.activeMaterial!,
          region: { x: x0, y: y0, width: w, height: h },
        });
        return [];
      });
      setMousePos(null);
    },
    [calibration.activeMaterial, toRaster, dispatch]
  );

  // Mouse move: update preview edge
  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!calibration.activeMaterial) return;
      const rect = overlayRef.current!.getBoundingClientRect();
      setMousePos([e.clientX - rect.left, e.clientY - rect.top]);
    },
    [calibration.activeMaterial]
  );

  // Escape: cancel current polygon
  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") { setDraftPts([]); setMousePos(null); }
    },
    []
  );

  useEffect(() => {
    const el = overlayRef.current;
    if (!el) return;
    if (!calibration.activeMaterial) {
      setDraftPts([]);
      setMousePos(null);
      return;
    }
    el.addEventListener("click", onClick);
    el.addEventListener("dblclick", onDblClick);
    el.addEventListener("mousemove", onMouseMove);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      el.removeEventListener("click", onClick);
      el.removeEventListener("dblclick", onDblClick);
      el.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [calibration.activeMaterial, onClick, onDblClick, onMouseMove, onKeyDown]);

  const activeColor = calibration.activeMaterial
    ? matColor(calibration.activeMaterial)
    : "#aaa";

  // Build SVG path string from screen points
  const svgPath = (pts: [number, number][], close = false) => {
    if (pts.length === 0) return "";
    const d = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
    return close ? d + " Z" : d;
  };

  // Existing region boxes converted to screen coords (re-derived on every viewKey change)
  const regionShapes = Object.values(calibration.swatches).flatMap((swatch) =>
    swatch.regions.map((reg, i) => {
      const [x0, y0] = toScreen(reg.x, reg.y);
      const [x1, y1] = toScreen(reg.x + reg.width, reg.y + reg.height);
      return { key: `${swatch.material}-${i}`, x0, y0, x1, y1, color: matColor(swatch.material) };
    })
  );

  return (
    <div
      ref={overlayRef}
      style={{
        position: "absolute", inset: 0, zIndex: 2000,
        cursor: calibration.activeMaterial ? "crosshair" : "default",
        pointerEvents: calibration.activeMaterial ? "auto" : "none",
      }}
    >
      {/* Status banner */}
      {calibration.activeMaterial && (
        <div style={{
          position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)",
          zIndex: 2001, padding: "4px 14px", borderRadius: 6,
          background: activeColor + "cc", color: "#fff", fontSize: 12, fontWeight: 600,
          pointerEvents: "none", whiteSpace: "nowrap",
        }}>
          {draftPts.length === 0
            ? `Click to start polygon — ${calibration.activeMaterial.replace("BM_", "")}`
            : `${draftPts.length} pts — double-click to finish  (Esc to cancel)`}
        </div>
      )}

      {/* SVG layer — all drawing */}
      <svg
        key={viewKey}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", overflow: "visible" }}
      >
        {/* Committed region bounding boxes */}
        {regionShapes.map(({ key, x0, y0, x1, y1, color }) => (
          <rect
            key={key}
            x={Math.min(x0, x1)} y={Math.min(y0, y1)}
            width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)}
            stroke={color} strokeWidth={2} fill={color + "33"} strokeDasharray="5 3"
          />
        ))}

        {/* Draft polygon fill */}
        {draftPts.length >= 3 && (
          <path
            d={svgPath(draftPts, true)}
            stroke={activeColor} strokeWidth={2} fill={activeColor + "22"} strokeDasharray="5 3"
          />
        )}

        {/* Draft polygon outline (open) */}
        {draftPts.length >= 2 && (
          <path
            d={svgPath(draftPts, false)}
            stroke={activeColor} strokeWidth={2} fill="none"
          />
        )}

        {/* Preview edge from last vertex to mouse */}
        {draftPts.length >= 1 && mousePos && (
          <line
            x1={draftPts[draftPts.length - 1][0]} y1={draftPts[draftPts.length - 1][1]}
            x2={mousePos[0]} y2={mousePos[1]}
            stroke={activeColor} strokeWidth={1.5} strokeDasharray="4 3" opacity={0.7}
          />
        )}

        {/* Vertex dots */}
        {draftPts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={4} fill={activeColor} stroke="#fff" strokeWidth={1.5} />
        ))}

        {/* Closing-snap circle on first vertex */}
        {draftPts.length >= 3 && (
          <circle cx={draftPts[0][0]} cy={draftPts[0][1]} r={8} fill="none" stroke={activeColor} strokeWidth={1} opacity={0.5} />
        )}
      </svg>
    </div>
  );
}
