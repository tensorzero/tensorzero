export default {
  test: {
    // Send results to Test Engine
    reporters: ["default", "buildkite-test-collector/vitest/reporter"],
    // Enable column + line capture for Test Engine
    includeTaskLocation: true,
  },
};
