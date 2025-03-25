import { test, expect } from "@playwright/test";

test("should show the dataset detail page", async ({ page }) => {
  await page.goto("/datasets/foo");
  await expect(page.getByText("Inference Type")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
