import { test, expect } from "@playwright/test";

/**
 * Tests for keyboard navigation in Combobox components.
 * These tests verify that keyboard navigation works correctly for comboboxes,
 * which is essential for the virtualized list implementation.
 */

test.describe("Combobox Keyboard Navigation", () => {
  test("function selector supports arrow key navigation and selection", async ({
    page,
  }) => {
    await page.goto("/playground");
    await expect(page.getByPlaceholder("Select function")).toBeVisible();

    // Open the function selector
    await page.getByPlaceholder("Select function").click();

    // Wait for dropdown to be visible
    await expect(page.getByRole("listbox")).toBeVisible();

    // Press ArrowDown to change selection
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");

    // Verify the third option is now selected (aria-selected)
    const options = page.getByRole("option");
    const thirdOption = options.nth(2);
    await expect(thirdOption).toHaveAttribute("aria-selected", "true");

    // Click the selected option to complete selection
    await thirdOption.click();

    // Wait for dropdown to close after selection
    await expect(page.getByRole("listbox")).not.toBeVisible();

    // Verify something was selected (combobox should have a value)
    await expect(page.getByPlaceholder("Select function")).not.toHaveValue("");
  });

  test("function selector supports Escape to close", async ({ page }) => {
    await page.goto("/playground");
    await expect(page.getByPlaceholder("Select function")).toBeVisible();

    // Open the function selector
    await page.getByPlaceholder("Select function").click();

    // Wait for dropdown to be visible
    await expect(page.getByRole("listbox")).toBeVisible();

    // Press Escape to close
    await page.keyboard.press("Escape");

    // Verify dropdown is closed
    await expect(page.getByRole("listbox")).not.toBeVisible();
  });
});
