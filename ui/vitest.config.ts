import path from "node:path";
import { fileURLToPath } from "node:url";
import { storybookTest } from "@storybook/addon-vitest/vitest-plugin";
import { playwright } from "@vitest/browser-playwright";
import { loadEnv } from "vite";
import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

const dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) =>
  mergeConfig(
    viteConfig({ mode, command: "serve" }),
    defineConfig({
      test: {
        projects: [
          // Unit tests project
          {
            extends: true,
            test: {
              name: "unit",
              environment: "node",
              include: ["**/*.test.ts", "**/*.test.tsx"],
              exclude: ["**/node_modules/**", "**/dist/**"],
              env: loadEnv(mode, process.cwd(), ""),
              reporters: [
                "default",
                "buildkite-test-collector/vitest/reporter",
                ["junit", { outputFile: "vitest.junit.xml" }],
              ],
              includeTaskLocation: true,
              deps: {
                optimizer: {
                  ssr: {
                    include: ["tslib"],
                  },
                },
              },
            },
          },
          // Storybook tests project
          {
            extends: true,
            plugins: [
              storybookTest({
                configDir: path.join(dirname, ".storybook"),
              }),
            ],
            test: {
              name: "storybook",
              browser: {
                enabled: true,
                headless: true,
                provider: playwright(),
                instances: [{ browser: "chromium" }],
              },
              setupFiles: ["./.storybook/vitest.setup.ts"],
            },
          },
        ],
      },
    }),
  ),
);
