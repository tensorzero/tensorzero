import { test, expect } from "@playwright/test";

test("should show the dataset builder page", async ({ page }) => {
  await page.goto("/datasets/builder");
  await expect(page.getByText("Create or update a dataset")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
