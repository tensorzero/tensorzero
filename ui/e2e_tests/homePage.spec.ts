import { test, expect } from "@playwright/test";

test("should show the home page", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByText(
      "Monitor metrics across models and prompts and debug individual API calls",
    ),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
