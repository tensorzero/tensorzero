import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    testTimeout: 30000, // 30 seconds for tests that may take longer
    // Send results to Test Engine
    reporters: ["default", "buildkite-test-collector/vitest/reporter"],
    // Enable column + line capture for Test Engine
    includeTaskLocation: true,
  },
});
