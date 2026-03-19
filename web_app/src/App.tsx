import { useEffect } from "react";
import { StoreProvider, useAppDispatch } from "./store";
import Layout from "./components/Layout";
import { fetchGpuInfo } from "./api/client";

function AppInit() {
  const dispatch = useAppDispatch();

  // Warn before closing/refreshing the window
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  useEffect(() => {
    // Fetch acceleration engine info; retry every 5 s until backend responds,
    // then re-check every 30 s in case the backend is restarted.
    let timer: ReturnType<typeof setTimeout>;

    const fetchAndSchedule = (delay: number) => {
      timer = setTimeout(async () => {
        try {
          const info = await fetchGpuInfo();
          if (info) {
            dispatch({
              type: "SET_ACCEL_INFO",
              info: { engine: info.engine, gpu: info.available, info: info.info },
            });
          }
          fetchAndSchedule(30_000); // re-check every 30 s
        } catch {
          fetchAndSchedule(5_000);  // backend not up yet — retry in 5 s
        }
      }, delay);
    };

    fetchAndSchedule(0); // immediate first fetch
    return () => clearTimeout(timer);
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
