import { test, expect } from "@playwright/test";

test("should show the function detail page", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByText("Variants")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
