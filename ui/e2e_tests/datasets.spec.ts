import { test, expect } from "@playwright/test";

test("should show the dataset list page", async ({ page }) => {
  await page.goto("/datasets");
  await expect(page.getByText("Dataset Name").first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
