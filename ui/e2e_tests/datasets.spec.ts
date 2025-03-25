import { test, expect } from "@playwright/test";

test("should show the dataset list page", async ({ page }) => {
  await page.goto("/datasets");
  await expect(page.getByText("Dataset Name")).toBeVisible();
  // TODO: remove
  await expect(page.getByText("Dataset Game")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
