import { test, expect } from "@playwright/test";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0193930b-6da0-7fa2-be87-9603d2bde664",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to edit and save a datapoint", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0195c49a-e011-7f60-a3a9-8c7f8fba2730",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Click the edit button
  await page.locator("button svg.lucide-pencil").click();

  // Edit the input
  await page.locator("textarea.font-mono").first().fill('{"topic":"foo"}');

  // Save the datapoint
  await page.locator("button svg.lucide-save").click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that the input is updated
  await expect(page.getByText('{"topic":"foo"}')).toBeVisible();
});
