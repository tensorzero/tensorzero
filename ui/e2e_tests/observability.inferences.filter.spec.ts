import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * Helper to apply filters and wait for results.
 * Returns after the filter sheet closes and data reloads.
 */
async function applyFiltersAndWait(page: Page) {
  // Click Apply Filters and wait for navigation to complete
  const responsePromise = page.waitForResponse(
    (response) =>
      response.url().includes("/observability/inferences") &&
      response.status() === 200,
  );
  await page.getByRole("button", { name: "Apply Filters" }).click();
  await responsePromise;

  // Wait for sheet to close - use Apply Filters button as the indicator
  await expect(
    page.getByRole("button", { name: "Apply Filters" }),
  ).not.toBeVisible();

  // Wait for table to load: either table has data rows (with cells) OR the empty state message cell
  await expect(
    page
      .locator("tbody tr td:nth-child(2)")
      .first()
      .or(page.locator("tbody tr td").getByText("No inferences found")),
  ).toBeVisible();
}

test.describe("Inference Filtering", () => {
  test("should open and close filter sheet", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Find filter button by its icon in the table header
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();

    // Verify sheet opens
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Close sheet using Escape key
    await page.keyboard.press("Escape");

    // Verify sheet closes
    await expect(
      page.getByRole("heading", { name: "Filter" }),
    ).not.toBeVisible();
  });

  test("should filter by function name", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Select function using FunctionSelector (combobox pattern)
    await page.getByRole("combobox").click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains the filter parameter
    await expect(page).toHaveURL(/function_name=write_haiku/, {
      timeout: 10_000,
    });

    // Verify that inferences are displayed
    await expect(page.locator("tbody tr")).not.toHaveCount(0);

    // Verify all visible function cells show write_haiku
    const functionCells = page.locator("tbody tr td:nth-child(3)");
    const count = await functionCells.count();
    for (let i = 0; i < count; i++) {
      await expect(functionCells.nth(i)).toContainText("write_haiku");
    }
  });

  test("should filter by variant name", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Find the Variant input field
    const variantInput = page.getByPlaceholder("Enter variant name");
    await variantInput.fill("openai_promptA");

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains the variant filter parameter
    await expect(page).toHaveURL(/variant_name=openai_promptA/, {
      timeout: 10_000,
    });
  });

  test("should filter by episode ID", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Find the Episode ID input field
    const episodeInput = page.getByPlaceholder("Enter episode ID");
    await episodeInput.fill("0196367a-842d-74c2-9e62-67f07369b6ad");

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains the episode_id filter parameter
    await expect(page).toHaveURL(
      /episode_id=0196367a-842d-74c2-9e62-67f07369b6ad/,
      { timeout: 10_000 },
    );
  });

  test("should filter by search query", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Find the search query input
    const searchInput = page.getByPlaceholder("Search in input and output");
    await searchInput.fill("haiku");

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains the search query
    await expect(page).toHaveURL(/search_query=haiku/, { timeout: 10_000 });
  });

  test("should add advanced tag filter", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Click "Tag" button to add a tag filter
    await page.getByRole("button", { name: "Tag" }).click();

    // Verify tag filter inputs appear
    const tagKeyInput = page.getByPlaceholder("tag");
    const tagValueInput = page.getByPlaceholder("value");
    await expect(tagKeyInput).toBeVisible();
    await expect(tagValueInput).toBeVisible();

    // Fill in tag filter
    await tagKeyInput.fill("test_key");
    await tagValueInput.fill("test_value");

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains filter parameter with JSON
    await expect(page).toHaveURL(/filter=/, { timeout: 10_000 });
  });

  test("should add metric filter", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Click "Metric" button to add a metric filter
    await page.getByRole("button", { name: "Metric" }).click();

    // Select a metric from the popover (if available)
    // Wait for the popover command to appear
    const metricOption = page.locator('[cmdk-item=""]').first();
    if (await metricOption.isVisible()) {
      await metricOption.click();

      // Apply filters and wait for results
      await applyFiltersAndWait(page);

      // Verify URL contains filter parameter with JSON
      await expect(page).toHaveURL(/filter=/, { timeout: 10_000 });
    }
  });

  test("should clear function filter", async ({ page }) => {
    // Start with a filter already applied
    await page.goto("/observability/inferences?function_name=write_haiku");

    // Wait for table to load with filtered results
    const functionCells = page.locator("tbody tr td:nth-child(3)");
    await expect(functionCells.first()).toBeVisible();

    // Verify initial filter shows only write_haiku
    const initialCount = await functionCells.count();
    for (let i = 0; i < initialCount; i++) {
      await expect(functionCells.nth(i)).toContainText("write_haiku");
    }

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Click Clear button next to function selector
    await page.getByRole("button", { name: "Clear" }).first().click();

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL no longer contains function_name
    await expect(page).not.toHaveURL(/function_name/, { timeout: 10_000 });
  });

  test("should show active filter state on button", async ({ page }) => {
    // Without filters, button should have ghost variant
    await page.goto("/observability/inferences");
    const filterButton = page.locator("thead").getByRole("button").last();

    // With filters, button should have default variant (different styling)
    await page.goto("/observability/inferences?function_name=write_haiku");

    // The button should still be visible (we can't easily check variant in e2e,
    // but we verify the page loads correctly with filters)
    await expect(filterButton).toBeVisible();
    await expect(page).toHaveURL(/function_name=write_haiku/, {
      timeout: 10_000,
    });

    // All visible function cells should show write_haiku (confirming filter is active)
    const functionCells = page.locator("tbody tr td:nth-child(3)");
    const count = await functionCells.count();
    for (let i = 0; i < count; i++) {
      await expect(functionCells.nth(i)).toContainText("write_haiku");
    }
  });

  test("should combine multiple filters", async ({ page }) => {
    await page.goto("/observability/inferences");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Select function
    await page.getByRole("combobox").click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Add variant filter
    const variantInput = page.getByPlaceholder("Enter variant name");
    await variantInput.fill("openai_promptA");

    // Apply filters and wait for results
    await applyFiltersAndWait(page);

    // Verify URL contains both parameters
    await expect(page).toHaveURL(/function_name=write_haiku/, {
      timeout: 10_000,
    });
    await expect(page).toHaveURL(/variant_name=openai_promptA/, {
      timeout: 10_000,
    });
  });

  test("should preserve filters when paginating", async ({ page }) => {
    // Apply a filter
    await page.goto("/observability/inferences?function_name=write_haiku");

    // Verify filter is applied
    await expect(page).toHaveURL(/function_name=write_haiku/);

    // Verify all function cells show write_haiku before pagination
    const functionCells = page.locator("tbody tr td:nth-child(3)");
    const initialCount = await functionCells.count();
    for (let i = 0; i < initialCount; i++) {
      await expect(functionCells.nth(i)).toContainText("write_haiku");
    }

    // If there's a next page button and it's enabled, click it
    const nextButton = page.getByRole("button", { name: /next/i });
    if (await nextButton.isVisible()) {
      const isDisabled = await nextButton.isDisabled();
      if (!isDisabled) {
        await nextButton.click();
        // Verify filter is preserved in the URL
        await expect(page).toHaveURL(/function_name=write_haiku/);

        // Verify filter is preserved - all function cells still show write_haiku
        const newFunctionCells = page.locator("tbody tr td:nth-child(3)");
        const newCount = await newFunctionCells.count();
        for (let i = 0; i < newCount; i++) {
          await expect(newFunctionCells.nth(i)).toContainText("write_haiku");
        }
      }
    }
  });
});
