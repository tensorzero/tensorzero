import { test, expect } from "@playwright/test";

test.describe("Error Boundaries", () => {
  test.describe("404 - Page Not Found", () => {
    test("should show error card for non-existent route", async ({ page }) => {
      await page.goto("/this-route-does-not-exist-12345");

      // Should show error card with 404 message
      await expect(page.getByText("Error 404")).toBeVisible();
      await expect(
        page.getByText("The requested page could not be found"),
      ).toBeVisible();

      // Sidebar should still be visible (not a full-page modal)
      await expect(page.getByText("Dashboard")).toBeVisible();
      await expect(page.getByText("Inferences")).toBeVisible();
    });

    test("should allow navigation away from 404 via sidebar", async ({
      page,
    }) => {
      await page.goto("/non-existent-page");

      // Verify we're on the error page
      await expect(page.getByText("Error 404")).toBeVisible();

      // Click on Dashboard in sidebar
      await page.getByRole("link", { name: "Dashboard" }).click();

      // Should navigate to home page successfully
      await expect(page.getByText("Ask a question")).toBeVisible();
      await expect(page.getByText("Error 404")).not.toBeVisible();
    });
  });

  test.describe("Resource Not Found", () => {
    test("should show error for non-existent inference", async ({ page }) => {
      // Navigate to an inference ID that doesn't exist
      await page.goto(
        "/observability/inferences/00000000-0000-0000-0000-000000000000",
      );

      // Should show an error (either 404 or "not found" message)
      const errorVisible = await page
        .getByText(/error|not found/i)
        .first()
        .isVisible()
        .catch(() => false);

      expect(errorVisible).toBe(true);

      // Sidebar should remain functional
      await expect(page.getByText("Inferences")).toBeVisible();
    });

    test("should show error for non-existent episode", async ({ page }) => {
      await page.goto(
        "/observability/episodes/00000000-0000-0000-0000-000000000000",
      );

      // Should show an error
      const errorVisible = await page
        .getByText(/error|not found/i)
        .first()
        .isVisible()
        .catch(() => false);

      expect(errorVisible).toBe(true);

      // Sidebar should remain functional
      await expect(page.getByText("Episodes")).toBeVisible();
    });

    test("should show error for non-existent dataset", async ({ page }) => {
      await page.goto("/datasets/this-dataset-does-not-exist-xyz");

      // Should show an error
      const errorVisible = await page
        .getByText(/error|not found/i)
        .first()
        .isVisible()
        .catch(() => false);

      expect(errorVisible).toBe(true);

      // Sidebar should remain functional
      await expect(page.getByText("Datasets")).toBeVisible();
    });

    test("should show error for non-existent function", async ({ page }) => {
      await page.goto("/observability/functions/this_function_does_not_exist");

      // Should show an error
      const errorVisible = await page
        .getByText(/error|not found/i)
        .first()
        .isVisible()
        .catch(() => false);

      expect(errorVisible).toBe(true);

      // Sidebar should remain functional
      await expect(page.getByText("Functions")).toBeVisible();
    });
  });

  test.describe("Error Card Display", () => {
    test("error card should have proper styling", async ({ page }) => {
      await page.goto("/non-existent-route");

      // Find the error card
      const errorCard = page.locator('[class*="border"]').filter({
        has: page.getByText("Error 404"),
      });

      await expect(errorCard).toBeVisible();

      // Card should have constrained width (max-w-lg = 512px)
      const cardBox = await errorCard.boundingBox();
      expect(cardBox).not.toBeNull();
      if (cardBox) {
        expect(cardBox.width).toBeLessThanOrEqual(600); // max-w-lg with some margin
      }
    });

    test("error card should be centered in content area", async ({ page }) => {
      await page.goto("/non-existent-route");

      const errorCard = page.locator('[class*="border"]').filter({
        has: page.getByText("Error 404"),
      });

      await expect(errorCard).toBeVisible();

      // Get viewport and card positions
      const viewportSize = page.viewportSize();
      const cardBox = await errorCard.boundingBox();

      expect(viewportSize).not.toBeNull();
      expect(cardBox).not.toBeNull();

      if (viewportSize && cardBox) {
        // Card should not be at the very left edge (should have some left offset for sidebar)
        expect(cardBox.x).toBeGreaterThan(50);
      }
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
      await expect(page.getByText("Error 404")).toBeVisible();

      // Navigate back using browser back button
      await page.goBack();

      // Should be back on valid page without errors
      await expect(page.getByText("Ask a question")).toBeVisible();
      await expect(page.getByText("Error 404")).not.toBeVisible();
    });

    test("should handle multiple error navigations gracefully", async ({
      page,
    }) => {
      // Navigate to multiple non-existent routes
      await page.goto("/invalid-1");
      await expect(page.getByText("Error 404")).toBeVisible();

      await page.goto("/invalid-2");
      await expect(page.getByText("Error 404")).toBeVisible();

      await page.goto("/invalid-3");
      await expect(page.getByText("Error 404")).toBeVisible();

      // Should still be able to navigate to valid page
      await page.goto("/");
      await expect(page.getByText("Ask a question")).toBeVisible();
    });
  });
});
