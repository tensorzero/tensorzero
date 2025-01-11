import { reactRouter } from "@react-router/dev/vite";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  css: {
    postcss: {
      plugins: [tailwindcss, autoprefixer],
    },
  },
  plugins: [wasm(), reactRouter(), tsconfigPaths()],
  // IMPORTANT:
  // If we don't set the target to es2022, we need `vite-plugin-top-level-await`
  // for "vite-plugin-wasm".
  // However, that's causing an error when building the production bundle.
  // For now, we're setting the target to es2022 as a workaround.
  build: {
    target: "es2022",
  },
});
