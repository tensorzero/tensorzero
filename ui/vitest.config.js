import { defineConfig } from "vitest/config";
import tsconfigPaths from "vite-tsconfig-paths";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  plugins: [tsconfigPaths(), wasm()],
  test: {
    environment: "node",
    include: ["**/*.test.ts", "**/*.test.tsx"],
    deps: {
      optimizer: {
        ssr: {
          include: ["tslib"],
        },
      },
    },
  },
});
