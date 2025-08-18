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

test("should show 17 functions in the functions badge", async ({ page }) => {
  await page.goto("/");

  // Find the functions card specifically by looking for the card that contains both "Functions" title and description
  const functionsCard = page.locator(".block").filter({
    has: page.locator('h3:has-text("Functions")'),
    hasText: "17 functions",
  });

  await expect(functionsCard).toBeVisible();
});
