import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

test("should be able to bulk add selected inferences to a new dataset from evaluations page", async ({
  page,
}) => {
  // Navigate to the evaluation page with a specific evaluation run
  await page.goto(
    "/evaluations/entity_extraction?evaluation_run_ids=0196374c-2b06-7f50-b187-80c15cec5a1f",
  );

  // Wait for the page to load completely
  await page.waitForLoadState("networkidle");

  // Verify the page loaded correctly
  await expect(page.getByText("Input")).toBeVisible();

  // Get all checkboxes in the table body (using role="checkbox" for Radix UI components)
  const checkboxes = page.locator('tbody button[role="checkbox"]');

  // Wait for checkboxes to be visible
  await checkboxes.first().waitFor({ state: "visible" });

  // Count total checkboxes available
  const checkboxCount = await checkboxes.count();
  expect(checkboxCount).toBeGreaterThanOrEqual(4);

  // Click the second checkbox (index 1)
  await checkboxes.nth(1).click();

  // Click the fourth checkbox (index 3)
  await checkboxes.nth(3).click();

  // Wait a moment for the state to update
  await page.waitForTimeout(300);

  // Verify the button shows the count of selected inferences
  const addButton = page.getByText(/Add.*2.*selected.*inferences.*to dataset/i);
  await expect(addButton).toBeVisible();

  // Generate a unique dataset name
  const datasetName = `test_bulk_eval_${v7()}`;

  // Click the add button to open the dataset selector
  await addButton.click();

  // Wait for the dropdown to appear
  await page.waitForTimeout(500);

  // Find the CommandInput by its placeholder text
  const commandInput = page.getByPlaceholder("Create or find a dataset...");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);

  // Wait a moment for the filtered results to appear
  await page.waitForTimeout(500);

  // Click on the CommandItem that contains the dataset name (create option)
  // Look for the "New dataset" group and click on the item within it
  const createOption = page
    .locator("[cmdk-item]")
    .filter({ hasText: datasetName });
  await createOption.waitFor({ state: "visible" });
  await createOption.click();

  // Wait for the toast to appear with success message
  const toastRegion = page.getByRole("region", { name: /notifications/i });
  await expect(toastRegion.getByText("Added to Dataset")).toBeVisible();

  // Verify the toast shows the correct count and dataset name
  await expect(
    toastRegion.getByText(
      new RegExp(`2.*inferences.*added to.*${datasetName}`),
    ),
  ).toBeVisible();

  // Wait for and click on the "View Dataset" button in the toast
  const viewButton = toastRegion.getByText("View Dataset");
  await viewButton.waitFor({ state: "visible" });
  await viewButton.click();

  // Wait for navigation to the dataset page
  await page.waitForURL(`/datasets/${datasetName}`, {
    timeout: 5000,
  });

  // Verify we're on the correct dataset page
  expect(page.url()).toContain(`/datasets/${datasetName}`);

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Verify we're on a dataset page by looking for the dataset name header
  await expect(page.getByText(datasetName).first()).toBeVisible();

  // Verify the dataset table shows 2 datapoints
  const datapointRows = page.locator("tbody tr");
  await expect(datapointRows).toHaveCount(2);

  // Click on the first datapoint's ID cell to view details (avoid clicking on function name links)
  const firstRowIdCell = datapointRows.first().locator("td").first();
  await firstRowIdCell.click();

  // Wait for navigation to the datapoint detail page
  await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
    timeout: 5000,
  });

  // Verify the datapoint detail page loaded correctly
  await expect(page.getByText("Datapoint", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();

  // Verify we can see some output content (the datapoints should have JSON output with person entities)
  // The entity_extraction function returns entities like "person", "organization", etc.
  // This confirms that the generated output (inherit mode) was properly saved to the dataset
  await expect(page.locator("body")).toContainText(/person|organization/i);
});
