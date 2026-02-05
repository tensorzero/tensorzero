import type { TestInfo } from "@playwright/test";

/**
 * Constructs a deterministic string based on the test title and retry number.
 *
 * This is the Playwright equivalent of the Rust `deterministic_test_and_attempt` function.
 * It enables prompt/model caching by ensuring the same test always produces the same name.
 *
 * The name includes:
 * - Test title (from Playwright's TestInfo)
 * - Retry attempt number (for test isolation on retries)
 * - A prefix for the specific resource type
 *
 * @param testInfo - Playwright's TestInfo object (available in test fixtures)
 * @param prefix - A deterministic prefix for the resource type
 * @returns A deterministic string unique to this test and attempt
 */
export function deterministicTestAndAttempt(
  testInfo: TestInfo,
  prefix: string,
): string {
  // Sanitize the test title to create a valid identifier
  // Replace spaces and special characters with underscores
  const sanitizedTitle = testInfo.title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  const attempt = testInfo.retry + 1; // retry is 0-indexed, make it 1-indexed like nextest

  return `test-${sanitizedTitle}-${attempt}-${prefix}`;
}

/**
 * Generate a unique dataset name for Playwright E2E tests.
 *
 * This is the Playwright equivalent of the Rust `unique_dataset_name` function.
 * The name is deterministic to enable prompt/model caching.
 *
 * @param testInfo - Playwright's TestInfo object
 * @param prefix - A deterministic prefix (e.g., "topk_viz")
 * @returns A deterministic dataset name unique to this test and attempt
 */
export function uniqueDatasetName(testInfo: TestInfo, prefix: string): string {
  return deterministicTestAndAttempt(testInfo, `dataset-${prefix}`);
}
