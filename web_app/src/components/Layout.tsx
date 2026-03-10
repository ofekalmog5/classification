import { useState } from "react";
import Sidebar from "./Sidebar";
import MapView from "./MapView";
import StatusBar from "./StatusBar";
import LayerPanel from "./LayerPanel";

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [layerPanelOpen, setLayerPanelOpen] = useState(true);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center h-11 px-4 bg-surface-900 border-b border-surface-700 shrink-0">
        <button
          onClick={() => setSidebarOpen((v) => !v)}
          className="mr-3 p-1 rounded hover:bg-surface-700 text-surface-300 transition-colors"
          title="Toggle sidebar"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <h1 className="text-sm font-semibold text-surface-200 tracking-wide select-none">
          Material Classification
        </h1>
        <div className="flex-1" />
        <button
          onClick={() => setLayerPanelOpen((v) => !v)}
          className="p-1 rounded hover:bg-surface-700 text-surface-300 transition-colors"
          title="Toggle layer panel"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="w-80 shrink-0 bg-surface-900 border-r border-surface-700 overflow-y-auto animate-slide-in">
            <Sidebar />
          </aside>
        )}

        {/* Map */}
        <main className="flex-1 relative">
          <MapView />
        </main>

        {/* Layer panel */}
        {layerPanelOpen && (
          <aside className="w-72 shrink-0 bg-surface-900 border-l border-surface-700 overflow-y-auto animate-slide-in">
            <LayerPanel />
          </aside>
        )}
      </div>

      {/* Status bar */}
      <StatusBar />
    </div>
  );
}
