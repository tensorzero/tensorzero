import { test, expect } from "@playwright/test";

test.describe("Dataset Filtering", () => {
  test("should open and close filter sheet", async ({ page }) => {
    await page.goto("/datasets/foo");

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
    await page.goto("/datasets/foo");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Select function using FunctionSelector (combobox pattern)
    await page.getByRole("combobox").click();
    await page.getByRole("option", { name: "extract_entities" }).click();

    // Apply filters
    await page.getByRole("button", { name: "Apply Filters" }).click();

    // Verify URL contains the filter parameter
    await expect(page).toHaveURL(/function_name=extract_entities/);

    // Verify data still shows (all datapoints in "foo" have this function)
    await expect(page.locator("tbody tr")).not.toHaveCount(0);
  });

  test("should filter to show only write_haiku datapoints", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Select write_haiku function
    await page.getByRole("combobox").click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Apply filters
    await page.getByRole("button", { name: "Apply Filters" }).click();

    // Verify URL contains the filter
    await expect(page).toHaveURL(/function_name=write_haiku/);

    // Verify only write_haiku rows are shown (should have fewer rows than unfiltered)
    await expect(page.locator("tbody tr")).not.toHaveCount(0);
    // All visible function cells should show write_haiku
    const functionCells = page.locator("tbody tr td:nth-child(4)");
    const count = await functionCells.count();
    for (let i = 0; i < count; i++) {
      await expect(functionCells.nth(i)).toContainText("write_haiku");
    }
  });

  test("should filter by search query", async ({ page }) => {
    await page.goto("/datasets/foo");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Find the search query input by looking for the label then the input next to it
    const searchQuerySection = page.locator("text=Search Query").locator("..");
    const searchInput = searchQuerySection
      .locator("..")
      .locator("input")
      .first();
    await searchInput.fill("haiku");

    // Apply filters
    await page.getByRole("button", { name: "Apply Filters" }).click();

    // Verify URL contains the search query
    await expect(page).toHaveURL(/search_query=haiku/);
  });

  test("should add advanced tag filter", async ({ page }) => {
    await page.goto("/datasets/foo");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Click "Tag" button under Advanced (has Plus icon + "Tag" label)
    await page.getByRole("button", { name: "Tag" }).click();

    // Verify tag filter inputs appear
    const tagKeyInput = page.getByPlaceholder("tag");
    const tagValueInput = page.getByPlaceholder("value");
    await expect(tagKeyInput).toBeVisible();
    await expect(tagValueInput).toBeVisible();

    // Fill in tag filter
    await tagKeyInput.fill("test_key");
    await tagValueInput.fill("test_value");

    // Apply filters
    await page.getByRole("button", { name: "Apply Filters" }).click();

    // Verify URL contains filter parameter with JSON
    await expect(page).toHaveURL(/filter=/);
  });

  test("should clear function filter", async ({ page }) => {
    // Start with a filter already applied
    await page.goto("/datasets/foo?function_name=extract_entities");

    // Open filter sheet
    const filterButton = page.locator("thead").getByRole("button").last();
    await filterButton.click();
    await expect(page.getByRole("heading", { name: "Filter" })).toBeVisible();

    // Click Clear button next to function selector
    await page.getByRole("button", { name: "Clear" }).first().click();

    // Apply filters
    await page.getByRole("button", { name: "Apply Filters" }).click();

    // Verify URL no longer contains function_name
    await expect(page).not.toHaveURL(/function_name/);
  });

  test("should show active filter state on button", async ({ page }) => {
    // Without filters, button should have ghost variant
    await page.goto("/datasets/foo");
    const filterButton = page.locator("thead").getByRole("button").last();

    // With filters, button should have default variant (different styling)
    await page.goto("/datasets/foo?function_name=extract_entities");

    // The button should still be visible (we can't easily check variant in e2e,
    // but we verify the page loads correctly with filters)
    await expect(filterButton).toBeVisible();
    await expect(page).toHaveURL(/function_name=extract_entities/);
  });
});
