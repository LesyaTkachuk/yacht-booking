import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ReactQueryClientProvider } from "./reactQueryProvider";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import "modern-normalize";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <ReactQueryClientProvider>
        <ReactQueryDevtools />
        <App />
      </ReactQueryClientProvider>
    </BrowserRouter>
  </StrictMode>
);
