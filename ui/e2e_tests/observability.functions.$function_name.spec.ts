import { test, expect } from "@playwright/test";

test("should show the function detail page", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByText("Variants")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should show description of chat function", async ({ page }) => {
  await page.goto("/observability/functions/write_haiku");
  await expect(page.getByText("Variants")).toBeVisible();
  await expect(
    page.getByText("Generate a haiku about a given topic"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should show description of json function", async ({ page }) => {
  await page.goto("/observability/functions/extract_entities");
  await expect(page.getByText("Variants")).toBeVisible();
  await expect(
    page.getByText("Extract named entities from text"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
