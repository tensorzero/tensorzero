import { test, expect } from "@playwright/test";

test.describe("Observability Models Page", () => {
  test("should display both charts and their toggle controls", async ({
    page,
  }) => {
    await page.goto("/observability/models");

    // Wait for the page to load - check for the main sections
    await expect(page.getByText("Model Usage Over Time")).toBeVisible();
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();

    // Check that Usage section is present
    await expect(page.getByRole("heading", { name: "Usage" })).toBeVisible();

    // Check that Latency section is present
    await expect(page.getByRole("heading", { name: "Latency" })).toBeVisible();

    // Verify that charts are rendered (look for chart containers)
    await expect(page.locator("[data-chart]")).toHaveCount(2);

    // Assert that "error" is not in the page
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should allow changing usage chart time granularity", async ({
    page,
  }) => {
    await page.goto("/observability/models");

    // Wait for the usage section to load
    await expect(page.getByText("Model Usage Over Time")).toBeVisible();

    // Find the time granularity selector for usage (first one on page)
    const usageTimeSelector = page.locator('button[role="combobox"]').first();
    await expect(usageTimeSelector).toBeVisible();

    // Click to open dropdown
    await usageTimeSelector.click();

    // Wait for dropdown to open and select a different time granularity
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Daily" }).click();

    // Verify the chart still displays after the change
    await expect(page.getByText("Model Usage Over Time")).toBeVisible();
    await expect(page.locator("[data-chart]").first()).toBeVisible();

    // Verify URL has been updated
    await expect(page).toHaveURL(/usageTimeGranularity=day/);
  });

  test("should allow changing usage chart metric type", async ({ page }) => {
    await page.goto("/observability/models");

    // Wait for the usage section to load
    await expect(page.getByText("Model Usage Over Time")).toBeVisible();

    // Find the metric selector for usage (second dropdown in the usage section)
    const metricSelector = page.locator('button[role="combobox"]').nth(1);
    await expect(metricSelector).toBeVisible();

    // Click to open dropdown
    await metricSelector.click();

    // Wait for dropdown to open and select a different metric
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Input Tokens" }).click();

    // Verify the chart still displays and description updates
    await expect(page.getByText("Input token usage by model")).toBeVisible();
    await expect(page.locator("[data-chart]").first()).toBeVisible();
  });

  test("should allow changing latency chart time granularity", async ({
    page,
  }) => {
    await page.goto("/observability/models");

    // Wait for the latency section to load
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();

    // Find the time granularity selector for latency (third dropdown on page)
    const latencyTimeSelector = page.locator('button[role="combobox"]').nth(2);
    await expect(latencyTimeSelector).toBeVisible();

    // Click to open dropdown
    await latencyTimeSelector.click();

    // Wait for dropdown to open and select a different time granularity
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Last Month" }).click();

    // Verify the chart still displays after the change
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();
    await expect(page.locator("[data-chart]").nth(1)).toBeVisible();

    // Verify URL has been updated
    await expect(page).toHaveURL(/latencyTimeGranularity=month/);
  });

  test("should allow changing latency chart metric type", async ({ page }) => {
    await page.goto("/observability/models");

    // Wait for the latency section to load
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();

    // Find the metric selector for latency (fourth dropdown on page)
    const latencyMetricSelector = page
      .locator('button[role="combobox"]')
      .nth(3);
    await expect(latencyMetricSelector).toBeVisible();

    // Click to open dropdown
    await latencyMetricSelector.click();

    // Wait for dropdown to open and select a different metric
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Time to First Token" }).click();

    // Verify the chart still displays after the change
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();
    await expect(page.locator("[data-chart]").nth(1)).toBeVisible();
  });

  test("should handle combinations of toggle changes", async ({ page }) => {
    await page.goto("/observability/models");

    // Wait for both sections to load
    await expect(page.getByText("Model Usage Over Time")).toBeVisible();
    await expect(page.getByText("Model Latency Distribution")).toBeVisible();

    // Change usage time granularity
    const usageTimeSelector = page.locator('button[role="combobox"]').first();
    await usageTimeSelector.click();
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Hourly" }).click();

    // Change usage metric
    const usageMetricSelector = page.locator('button[role="combobox"]').nth(1);
    await usageMetricSelector.click();
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Total Tokens" }).click();

    // Change latency time granularity
    const latencyTimeSelector = page.locator('button[role="combobox"]').nth(2);
    await latencyTimeSelector.click();
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "All Time" }).click();

    // Change latency metric
    const latencyMetricSelector = page
      .locator('button[role="combobox"]')
      .nth(3);
    await latencyMetricSelector.click();
    await page.waitForSelector('[role="option"]');
    await page.getByRole("option", { name: "Time to First Token" }).click();

    // Verify both charts still display
    await expect(page.locator("[data-chart]")).toHaveCount(2);
    await expect(
      page.getByText("Total token usage (input + output) by model"),
    ).toBeVisible();

    // Verify URL contains both parameters
    await expect(page).toHaveURL(/usageTimeGranularity=hour/);
    await expect(page).toHaveURL(/latencyTimeGranularity=cumulative/);
  });
});
