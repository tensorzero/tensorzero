import { test, expect } from "@playwright/test";
import { createDatapointFromInference } from "./helpers/datapoint-helpers";

test.describe("Dataset Deletion", () => {
  test("should delete individual datapoint with confirmation dialog", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createDatapointFromInference(page);

    await page.goto(`/datasets/${datasetName}`);

    // Wait for the page to load
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Get the first datapoint row
    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();

    // Click the delete button in the last column
    const deleteButton = firstRow.locator("td").last().locator("button");
    await expect(deleteButton).toBeVisible();
    await deleteButton.click();

    // Wait for the confirmation dialog
    await expect(page.getByText("Delete Datapoint")).toBeVisible();
    await expect(
      page.getByText(
        "The datapoint will be marked as stale in the database (soft deletion). This action cannot be undone.",
      ),
    ).toBeVisible();

    // Click "Delete" to confirm
    await page.getByRole("button", { name: "Delete" }).click();

    // Wait for the dialog to close and page to reload
    await expect(page.getByText("Delete Datapoint")).not.toBeVisible();
    await page.waitForLoadState("networkidle");

    // Verify the deleted datapoint is no longer in the table
    // await expect(firstRow).not.toBeVisible();
    await expect(page).toHaveURL(/\/datasets\/?$/, { timeout: 15000 });
    await expect(page.getByText(datasetName)).toHaveCount(0, {
      timeout: 10000,
    });
  });

  test("should cancel datapoint deletion when clicking cancel", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createDatapointFromInference(page);

    await page.goto(`/datasets/${datasetName}`);

    // Wait for the page to load
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Get the first datapoint row
    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();

    // Click the delete button
    const deleteButton = firstRow.locator("td").last().locator("button");
    await expect(deleteButton).toBeVisible();
    await deleteButton.click();

    // Wait for the confirmation dialog
    await expect(page.getByText("Delete Datapoint")).toBeVisible();

    // Click "Cancel" instead of "Delete"
    await page.getByRole("button", { name: "Cancel" }).click();

    // Verify the dialog is closed and datapoint still exists
    await expect(page.getByText("Delete Datapoint")).not.toBeVisible();
    await expect(firstRow).toBeVisible();
  });

  test("should delete entire dataset with confirmation buttons", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createDatapointFromInference(page);

    await page.goto(`/datasets/${datasetName}`);

    // Wait for the page to load
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Click the dataset delete button (in the page header, not in the table)
    const datasetDeleteButton = page
      .getByRole("button", { name: "Delete", exact: true })
      .filter({ hasNot: page.locator("tbody tr") });

    await expect(datasetDeleteButton).toBeVisible();
    await datasetDeleteButton.click();

    // Wait for the confirmation buttons to appear
    await expect(page.getByText("No, keep it")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("Yes, delete permanently")).toBeVisible({
      timeout: 10000,
    });

    // Click "Yes, delete permanently"
    await page.getByText("Yes, delete permanently").click();

    // Verify we're redirected to the datasets list page
    await expect(page).toHaveURL("/datasets");
    await expect(page.getByText("Dataset Name")).toBeVisible();
  });

  test("should cancel dataset deletion when clicking 'No, keep it'", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createDatapointFromInference(page);

    await page.goto(`/datasets/${datasetName}`);

    // Wait for the page to load
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Click the dataset delete button
    const datasetDeleteButton = page
      .getByRole("button", { name: "Delete", exact: true })
      .filter({ hasNot: page.locator("tbody tr") });

    await expect(datasetDeleteButton).toBeVisible();
    await datasetDeleteButton.click();

    // Wait for the confirmation buttons to appear
    await expect(page.getByText("No, keep it")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("Yes, delete permanently")).toBeVisible({
      timeout: 10000,
    });

    // Click "No, keep it" to cancel
    await page.getByText("No, keep it").click();

    // Verify we're still on the dataset page
    await expect(page).toHaveURL(`/datasets/${datasetName}`);
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Verify the delete button is back to its original state
    await expect(datasetDeleteButton).toBeVisible();
  });

  test("should redirect to datasets list when deleting the last datapoint", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createDatapointFromInference(page);

    await page.goto(`/datasets/${datasetName}`);

    // Wait for the page to load
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible();
    await expect(page.getByText(datasetName).nth(1)).toBeVisible();

    // Verify there's only one datapoint in the table
    const rows = page.locator("tbody tr");
    await expect(rows).toHaveCount(1);

    // Get the single datapoint row
    const firstRow = rows.first();
    await expect(firstRow).toBeVisible();

    // Click the delete button in the last column
    const deleteButton = firstRow.locator("td").last().locator("button");
    await expect(deleteButton).toBeVisible();
    await deleteButton.click();

    // Wait for the confirmation dialog
    await expect(page.getByText("Delete Datapoint")).toBeVisible();

    // Click "Delete" to confirm
    await page.getByRole("button", { name: "Delete" }).click();

    // Wait for redirect to datasets list page
    await expect(page).toHaveURL("/datasets", { timeout: 10000 });
    await expect(page.getByText("Dataset Name")).toBeVisible();

    // Verify the dataset is no longer in the list (since it's now empty)
    await expect(page.getByText(datasetName)).not.toBeVisible();
  });
});
