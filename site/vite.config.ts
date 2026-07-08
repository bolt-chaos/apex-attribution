import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages project path (https://<user>.github.io/apex-attribution/).
// In dev, BASE_URL resolves to "/"; data is fetched as `${import.meta.env.BASE_URL}data/*.json`.
export default defineConfig({
  base: "/apex-attribution/",
  plugins: [react()],
});
