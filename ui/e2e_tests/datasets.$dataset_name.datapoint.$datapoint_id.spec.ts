import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

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
    "/datasets/foo/datapoint/0193514c-ec40-7911-ad63-460bb9c861e1",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("Input")).toBeVisible();

  // Click the edit button
  await page.locator("button svg.lucide-pencil").click();

  // Edit the input
  const topic = v7();
  const input = `{"topic":"${topic}"}`;

  await page.locator("textarea.font-mono").first().fill(input);

  // Save the datapoint
  await page.locator("button svg.lucide-save").click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that the input is updated
  await expect(page.getByText(input)).toBeVisible();
  // NOTE: there will now be a new datapoint ID for this datapoint as its input has been edited
});
