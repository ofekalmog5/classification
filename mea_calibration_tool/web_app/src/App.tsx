import { AppProvider } from "./store";
import MapView from "./components/MapView";
import CalibrationSidebar from "./components/CalibrationSidebar";

export default function App() {
  return (
    <AppProvider>
      <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
        <CalibrationSidebar />
        <MapView />
      </div>
    </AppProvider>
  );
}
