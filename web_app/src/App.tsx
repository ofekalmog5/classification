import { StoreProvider } from "./store";
import Layout from "./components/Layout";

export default function App() {
  return (
    <StoreProvider>
      <Layout />
    </StoreProvider>
  );
}
