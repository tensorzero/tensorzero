import { test, expect } from "@playwright/test";

test("should show the DICL variant detail page", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities/variants/dicl");

  // Verify DICL-specific fields
  await expect(page.getByText("k (Neighbors)")).toBeVisible();
  await expect(page.getByText("Max Distance")).toBeVisible();
  const maxDistanceRow = page.getByText("Max Distance").locator("..");
  await expect(maxDistanceRow.getByText("0.5", { exact: true })).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
