import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: Vite ở :5173, chuyển /api và /ws sang FastAPI ở :8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
