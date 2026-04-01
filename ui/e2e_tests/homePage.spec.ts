import { test, expect } from "@playwright/test";

test("should show the home page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Ask a question")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("@base-path should check the health endpoint with the correct base path", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("TensorZero Gateway")).toBeVisible();
});

test("should show 22 functions in the functions badge", async ({ page }) => {
  await page.goto("/");

  // Find the functions card by looking for a link that contains "Functions" heading and "22 functions" text
  const functionsCard = page.locator('a[href*="functions"]').filter({
    has: page.locator('h3:has-text("Functions")'),
    hasText: "22 functions",
  });

  await expect(functionsCard).toBeVisible();
});

test("should show the models badge", async ({ page }) => {
  await page.goto("/");

  // Find the models card by looking for a link that contains "Models" heading and "models used" text
  const modelsCard = page.locator('a[href*="models"]').filter({
    has: page.locator('h3:has-text("Models")'),
    hasText: "models used",
  });

  await expect(modelsCard).toBeVisible();
});
