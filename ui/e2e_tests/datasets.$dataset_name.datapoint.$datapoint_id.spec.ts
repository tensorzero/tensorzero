import { test, expect } from "@playwright/test";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0193930b-6da0-7fa2-be87-9603d2bde664",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
