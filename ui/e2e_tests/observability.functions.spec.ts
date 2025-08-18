import { test, expect } from "@playwright/test";

test("should show the function list page", async ({ page }) => {
  await page.goto("/observability/functions");
  await expect(page.getByText("Variants")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
