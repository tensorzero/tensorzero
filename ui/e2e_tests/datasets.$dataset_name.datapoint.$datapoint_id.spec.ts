import { test, expect } from "@playwright/test";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0195c49a-e011-7f60-a3a9-8c7f8fba2730",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
