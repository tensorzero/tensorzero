import { test, expect } from "@playwright/test";

test("should show the episode list page", async ({ page }) => {
  await page.goto("/observability/episodes");
  await expect(page.getByText("Episode ID")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
