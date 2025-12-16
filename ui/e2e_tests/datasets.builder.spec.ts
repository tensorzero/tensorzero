import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * Generate unique dataset name for each test to avoid conflicts
 */
function getTestDatasetName(testName: string) {
  return `e2e_${testName}_${Date.now()}`;
}

/**
 * Helper to wait for dataset creation to complete and redirect.
 */
async function createDatasetAndWait(page: Page) {
  // Wait for navigation to dataset page after submission
  const responsePromise = page.waitForURL(/\/datasets\/.*\?rowsAdded=/, {
    timeout: 30_000,
  });
  // Use type="submit" selector for stability - button text may vary
  await page.locator('button[type="submit"]').click();
  await responsePromise;
}

test.describe("Dataset Builder", () => {
  test("should show the dataset builder page", async ({ page }) => {
    await page.goto("/datasets/builder");
    await expect(page.getByText("Outputs to be used in dataset")).toBeVisible();

    // Assert that "error" is not in the page
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should create dataset from function", async ({ page }) => {
    const datasetName = getTestDatasetName("basic");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page with rowsAdded param
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist in the table
    await expect(page.locator("tbody tr").first()).toBeVisible();
  });

  test("should create dataset filtered by variant", async ({ page }) => {
    const datasetName = getTestDatasetName("variant");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Wait for count to load and capture unfiltered count
    const countText = page.getByText(/\d+ matching inferences/);
    await expect(countText).toBeVisible();
    const unfilteredText = await countText.textContent();
    const unfilteredCount = parseInt(
      unfilteredText?.match(/(\d+)/)?.[1] || "0",
    );

    // Select variant (wait for it to be enabled after function selection)
    const variantCombobox = page.getByRole("combobox", { name: "Variant" });
    await expect(variantCombobox).toBeEnabled();
    await variantCombobox.click();
    await page
      .getByRole("option", { name: "initial_prompt_gpt4o_mini" })
      .click();

    // Wait for count to update after variant selection
    // The count should be less than before (variant filter reduces results)
    await expect(async () => {
      const newText = await countText.textContent();
      const filteredCount = parseInt(newText?.match(/(\d+)/)?.[1] || "0");
      expect(filteredCount).toBeLessThan(unfilteredCount);
      expect(filteredCount).toBeGreaterThan(0);
    }).toPass({ timeout: 5000 });

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist
    await expect(page.locator("tbody tr").first()).toBeVisible();
  });

  test("should create dataset filtered by tag", async ({ page }) => {
    const datasetName = getTestDatasetName("tag");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function (answer_question has tags foo=bar)
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "answer_question" }).click();

    // Add tag filter
    await page.getByRole("button", { name: "Tag" }).click();
    await page.getByPlaceholder("tag").fill("foo");
    await page.getByPlaceholder("value").fill("bar");

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist
    await expect(page.locator("tbody tr").first()).toBeVisible();
  });

  test("should create dataset filtered by float metric", async ({ page }) => {
    const datasetName = getTestDatasetName("float_metric");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Add metric filter: haiku_rating >= 0.5
    await page.getByRole("button", { name: "Metric" }).click();
    await page.locator('[data-value="haiku_rating"]').click();

    // Change operator to >= and value to 0.5
    await page
      .locator('button[aria-label="Metric comparison operator"]')
      .click();
    await page.getByRole("option", { name: "≥" }).click();
    await page.getByRole("spinbutton").fill("0.5");

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist
    await expect(page.locator("tbody tr").first()).toBeVisible();
  });

  test("should create dataset filtered by boolean metric", async ({ page }) => {
    const datasetName = getTestDatasetName("bool_metric");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Add metric filter: haiku_score = true
    await page.getByRole("button", { name: "Metric" }).click();
    await page.locator('[data-value="haiku_score"]').click();

    // Boolean metric defaults to true, so we don't need to change it

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist
    await expect(page.locator("tbody tr").first()).toBeVisible();
  });

  test("should create dataset filtered by search query", async ({ page }) => {
    const datasetName = getTestDatasetName("search");

    await page.goto("/datasets/builder");

    // Select new dataset name
    await page.getByRole("combobox", { name: "Dataset" }).click();
    await page.getByPlaceholder(/Create or find a dataset/).fill(datasetName);
    await page.getByRole("option", { name: datasetName }).click();

    // Select function
    await page.getByRole("combobox", { name: "Function" }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Enter search query
    await page.getByPlaceholder("Search in input and output").fill("mouton");

    // Create dataset
    await createDatasetAndWait(page);

    // Verify we're on the dataset page with results
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}\\?rowsAdded=\\d+`),
    );

    // Verify datapoints exist (search filter worked)
    await expect(page.locator("tbody tr").first()).toBeVisible();

    // Click on first datapoint ID to navigate to datapoint page
    await page.locator("tbody tr td:first-child a").first().click();

    // Verify we're on a datapoint page
    await expect(page).toHaveURL(/\/datapoint\//);

    // Verify "mouton" appears somewhere in the page content (may be in JSON/code block)
    await expect(page.locator("text=mouton").first()).toBeVisible();
  });
});
