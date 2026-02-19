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

test("should show the default function variant detail page", async ({
  page,
}) => {
  await page.goto(
    "/observability/functions/tensorzero%3A%3Adefault/variants/openai%3A%3Agpt-4o-mini",
  );

  // Verify the page loaded with expected content
  await expect(page.getByText("Variants").first()).toBeVisible();
  await expect(page.getByText("openai::gpt-4o-mini").first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  await expect(page.getByText("404", { exact: false })).not.toBeVisible();
});
