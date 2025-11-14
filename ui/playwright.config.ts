import "dotenv/config";

import { defineConfig, devices } from "@playwright/test";

const useUIDocker =
  process.env.TENSORZERO_PLAYWRIGHT_NO_WEBSERVER || process.env.TENSORZERO_CI;
// Allow docker-compose to override baseURL explicitly (e.g., http://ui:4000 in CI)
const baseURLOverride = process.env.TENSORZERO_PLAYWRIGHT_BASE_URL;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: "./e2e_tests",
  /* Global timeout: 60 seconds max per test */
  timeout: 60000,
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.TENSORZERO_CI,
  /* Retry on CI only */
  retries: process.env.TENSORZERO_PLAYWRIGHT_RETRIES
    ? parseInt(process.env.TENSORZERO_PLAYWRIGHT_RETRIES)
    : process.env.TENSORZERO_CI
      ? 2
      : 0,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: process.env.TENSORZERO_CI
    ? [
        ["list"],
        ["github"],
        ["buildkite-test-collector/playwright/reporter"],
        ["junit", { outputFile: "playwright.junit.xml" }],
      ]
    : [["dot"], ["junit", { outputFile: "playwright.junit.xml" }]],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL:
      baseURLOverride ||
      (useUIDocker ? "http://localhost:4000" : "http://localhost:5173"),

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on",
    // video: "on-first-retry",
    video: "on",
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1920, height: 1080 },
      },
    },

    // {
    //   name: "firefox",
    //   use: { ...devices["Desktop Firefox"] },
    // },

    // {
    //   name: "webkit",
    //   use: { ...devices["Desktop Safari"] },
    // },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: { ...devices['Desktop Edge'], channel: 'msedge' },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ],

  /* Run your local dev server before starting the tests if not in CI */
  webServer: useUIDocker
    ? undefined
    : {
        command: "pnpm run dev",
        url: "http://localhost:5173",
        reuseExistingServer: true,
        stdout: "pipe",
        stderr: "pipe",
      },
});
