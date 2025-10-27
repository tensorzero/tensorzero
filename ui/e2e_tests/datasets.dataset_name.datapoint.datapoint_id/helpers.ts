import type { Page } from "@playwright/test";

/**
 * Expands all "Show more" buttons to prevent gradient overlay from blocking clicks.
 *
 * The UI uses a gradient overlay at the bottom of collapsed sections with a "Show more" button.
 * This overlay can intercept pointer events and prevent clicks on buttons below it.
 * Each message section has at most ONE "Show more" button - clicking it once reveals all content.
 *
 * This function waits briefly for buttons to appear (content needs time to render and determine
 * if it should be collapsed), then clicks ALL of them to expand all collapsed content.
 *
 * @param page - The Playwright Page object
 *
 * @example
 * ```ts
 * await page.getByRole("button", { name: "Edit" }).click();
 * await expandShowMoreIfPresent(page);
 * // Or scope to a specific section:
 * const section = page.getByTestId("message-assistant");
 * await expandShowMoreIfPresent(section);
 * ```
 */
export async function expandShowMoreIfPresent(page: Page): Promise<void> {
  // Wait briefly for any show more buttons to appear
  const showMoreButtons = page.getByRole("button", {
    name: "Show more",
  });

  try {
    // Keep clicking show more buttons until none remain
    while (true) {
      const button = showMoreButtons.first();
      await button.waitFor({ state: "visible", timeout: 200 });
      await button.click();
    }
  } catch {
    // No more buttons - all content expanded
    return;
  }
}

/**
 * Clicks the Save button and waits for the page to redirect and fully load.
 *
 * This helper automates the common pattern of:
 * 1. Capture current URL
 * 2. Click the "Save" button
 * 3. Wait for URL to change (navigation completes)
 * 4. Wait for network activity to settle (networkidle)
 * 5. Wait for the "Edit" button to appear (confirms page is fully rendered in read-only mode)
 *
 * Use this after making edits to a datapoint to ensure the save completes
 * and the page is ready for assertions.
 *
 * @param page - The Playwright Page object
 *
 * @example
 * ```ts
 * // Make some edits
 * await page.getByRole("button", { name: "Edit" }).click();
 * await someEditor.fill("new content");
 *
 * // Save and wait for redirect
 * await saveAndWaitForRedirect(page);
 *
 * // Now safe to verify content
 * await expect(page.getByText("new content")).toBeVisible();
 * ```
 */
export async function saveAndWaitForRedirect(page: Page): Promise<void> {
  const currentUrl = page.url();
  await page.getByRole("button", { name: "Save" }).click();
  await page.waitForURL((url) => url.toString() !== currentUrl, {
    timeout: 5000,
  });
  await page.waitForLoadState("networkidle", { timeout: 5000 });
  await page
    .getByRole("button", { name: "Edit" })
    .waitFor({ state: "visible" });
}
