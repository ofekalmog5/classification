import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Warn before closing/refreshing the page to prevent accidental loss of work
window.addEventListener("beforeunload", (e) => {
  e.preventDefault();
  // Modern browsers show a generic message; the returnValue assignment is
  // required for the prompt to appear in Chrome / Edge.
  e.returnValue = "";
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
