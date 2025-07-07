import { test, expect } from "@playwright/test";

test("should show the variant detail page", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities/variants/dicl");
  // Weight only will show up if it is set in the config
  await expect(page.getByText("Weight")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
