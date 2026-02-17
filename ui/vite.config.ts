import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, loadEnv } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import devtoolsJson from "vite-plugin-devtools-json";
import wasm from "vite-plugin-wasm";
import react from "@vitejs/plugin-react";

// We don't need to load `reactRouter` in storybook or tests,
// but we need to load `react`.
const shouldLoadReactRouter =
  !process.env.VITEST && !process.argv[1]?.includes("storybook");

export default defineConfig(({ mode }) => ({
  plugins: [
    devtoolsJson(),
    wasm(),
    tailwindcss(),
    shouldLoadReactRouter ? reactRouter() : react(),
    tsconfigPaths(),
  ],

  define: {
    __APP_VERSION__: JSON.stringify(
      process.env.npm_package_version || "unknown",
    ),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },

  // IMPORTANT:
  // If we don't set the target to es2022, we need `vite-plugin-top-level-await`
  // for "vite-plugin-wasm".
  // However, that's causing an error when building the production bundle.
  // For now, we're setting the target to es2022 as a workaround.
  build: {
    target: "es2022",
  },
  server: shouldLoadReactRouter
    ? // This should fix a bug in React Router that causes the dev server to crash
      // on the first page load after clearing node_modules. Remove this when the
      // issue is fixed.
      // https://github.com/remix-run/react-router/issues/12786#issuecomment-2634033513
      { warmup: { clientFiles: ["./app/root.tsx"] } }
    : undefined,
  test: {
    // Load env variables from .env if it exists
    // https://vite.dev/config/
    env: loadEnv(mode, process.cwd(), ""),
    environment: "node",
    include: ["**/*.test.ts", "**/*.test.tsx"],
    reporters: ["default", ["junit", { outputFile: "vitest.junit.xml" }]],
    // Enable column + line capture for Test Engine
    includeTaskLocation: true,
    deps: {
      optimizer: {
        ssr: {
          include: ["tslib"],
        },
      },
    },
  },
}));
