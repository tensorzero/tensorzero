import { test, expect } from "@playwright/test";

test("should show the supervised fine-tuning page", async ({ page }) => {
  await page.goto("/optimization/supervised-fine-tuning");
  await expect(page.getByText("Advanced Parameters")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
