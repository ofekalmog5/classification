import { useEffect } from "react";
import { StoreProvider, useAppDispatch } from "./store";
import Layout from "./components/Layout";
import { fetchGpuInfo } from "./api/client";

function AppInit() {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Check acceleration engine as soon as the app loads
    fetchGpuInfo().then((info) => {
      if (info) {
        dispatch({
          type: "SET_ACCEL_INFO",
          info: { engine: info.engine, gpu: info.available, info: info.info },
        });
      }
    });
  }, [dispatch]);

  return <Layout />;
}

export default function App() {
  return (
    <StoreProvider>
      <AppInit />
    </StoreProvider>
  );
}
