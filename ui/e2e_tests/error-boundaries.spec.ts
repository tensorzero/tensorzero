import { test, expect } from "@playwright/test";

test.describe("Error Boundaries", () => {
  test.describe("404 - Page Not Found", () => {
    test("should show 404 display for non-existent route", async ({ page }) => {
      await page.goto("/this-route-does-not-exist-12345");

      // Should show the NotFoundDisplay with muted styling
      await expect(page.getByText("Page Not Found")).toBeVisible();
      await expect(
        page.getByText("The page you're looking for doesn't exist"),
      ).toBeVisible();

      // Sidebar should still be visible and functional
      await expect(page.getByText("Dashboard")).toBeVisible();
      await expect(page.getByText("Inferences")).toBeVisible();
    });

    test("should allow navigation away from 404 via sidebar", async ({
      page,
    }) => {
      await page.goto("/non-existent-page");

      // Verify we're on the error page
      await expect(page.getByText("Page Not Found")).toBeVisible();

      // Click on Dashboard in sidebar
      await page.getByRole("link", { name: "Dashboard" }).click();

      // Should navigate to home page successfully
      await expect(page.getByText("Ask a question")).toBeVisible();
      await expect(page.getByText("Page Not Found")).not.toBeVisible();
    });
  });

  test.describe("Navigation Recovery", () => {
    test("should recover from error state when navigating to valid page", async ({
      page,
    }) => {
      // Start on a valid page
      await page.goto("/");
      await expect(page.getByText("Ask a question")).toBeVisible();

      // Navigate to error page
      await page.goto("/invalid-route");
      await expect(page.getByText("Page Not Found")).toBeVisible();

      // Navigate back using browser back button
      await page.goBack();

      // Should be back on valid page without errors
      await expect(page.getByText("Ask a question")).toBeVisible();
      await expect(page.getByText("Page Not Found")).not.toBeVisible();
    });

    test("should handle multiple error navigations gracefully", async ({
      page,
    }) => {
      // Navigate to multiple non-existent routes
      await page.goto("/invalid-1");
      await expect(page.getByText("Page Not Found")).toBeVisible();

      await page.goto("/invalid-2");
      await expect(page.getByText("Page Not Found")).toBeVisible();

      await page.goto("/invalid-3");
      await expect(page.getByText("Page Not Found")).toBeVisible();

      // Should still be able to navigate to valid page
      await page.goto("/");
      await expect(page.getByText("Ask a question")).toBeVisible();
    });
  });
});
