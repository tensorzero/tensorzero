import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import wasm from "vite-plugin-wasm";
import react from "@vitejs/plugin-react";

// We don't need to load `reactRouter` in storybook or tests,
// but we need to load `react`.
const shouldLoadReactRouter =
  !process.env.VITEST && !process.argv[1]?.includes("storybook");

export default defineConfig({
  plugins: [
    wasm(),
    tailwindcss(),
    shouldLoadReactRouter ? reactRouter() : react(),
    tsconfigPaths(),
  ],
  // IMPORTANT:
  // If we don't set the target to es2022, we need `vite-plugin-top-level-await`
  // for "vite-plugin-wasm".
  // However, that's causing an error when building the production bundle.
  // For now, we're setting the target to es2022 as a workaround.
  build: {
    target: "es2022",
  },
});
