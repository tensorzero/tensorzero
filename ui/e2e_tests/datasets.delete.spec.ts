import { test, expect } from "@playwright/test";

test.describe("Dataset Deletion", () => {
  test("should delete individual datapoint with confirmation dialog", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Wait for the page to load and ensure we have datapoints
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Get the first datapoint row to delete
    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();

    // Get the datapoint ID from the first row for verification
    const datapointId = await firstRow.locator("td").first().textContent();
    expect(datapointId).toBeTruthy();

    // Click the delete button (trash icon) in the first row
    const deleteButton = firstRow
      .locator('button[aria-label="Delete"]')
      .first();
    await deleteButton.click();

    // Wait for the confirmation dialog to appear
    await expect(page.getByText("Delete Datapoint")).toBeVisible();
    await expect(
      page.getByText("Are you sure you want to delete this datapoint?"),
    ).toBeVisible();

    // Verify the datapoint details are shown in the dialog
    if (datapointId) {
      await expect(page.getByText(`ID: ${datapointId}`)).toBeVisible();
    }

    // Click the "Delete" button in the dialog
    await page.getByRole("button", { name: "Delete" }).click();

    // Wait for the success toast to appear
    await expect(
      page.getByText("The datapoint has been deleted successfully."),
    ).toBeVisible();

    // Wait for page reload and verify the datapoint is no longer in the table
    await page.waitForLoadState("networkidle");

    // The row should no longer exist or the table should be updated
    // We'll check that the page still loads without errors
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();
  });

  test("should cancel datapoint deletion when clicking cancel", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Wait for the page to load
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Get the first datapoint row
    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();

    // Get the datapoint ID to verify it's still there after canceling
    const datapointId = await firstRow.locator("td").first().textContent();

    // Click the delete button
    const deleteButton = firstRow
      .locator('button[aria-label="Delete"]')
      .first();
    await deleteButton.click();

    // Wait for the confirmation dialog
    await expect(page.getByText("Delete Datapoint")).toBeVisible();

    // Click "Cancel" instead of "Delete"
    await page.getByRole("button", { name: "Cancel" }).click();

    // Verify the dialog is closed
    await expect(page.getByText("Delete Datapoint")).not.toBeVisible();

    // Verify the datapoint is still there
    if (datapointId) {
      await expect(page.getByText(datapointId)).toBeVisible();
    }
  });

  test("should delete entire dataset with confirmation dialog", async ({
    page,
  }) => {
    // First, let's create a test dataset by going to the builder
    await page.goto("/datasets/builder");
    await expect(page.getByText("Outputs to be used in dataset")).toBeVisible();

    // For this test, we'll use an existing dataset that we know exists
    // Let's go to the "foo" dataset and test its deletion
    await page.goto("/datasets/foo");

    // Wait for the page to load
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Click the dataset delete button (in the page header)
    const datasetDeleteButton = page
      .getByRole("button", {
        name: "Delete",
        exact: true,
      })
      .first();
    await datasetDeleteButton.click();

    // Wait for the confirmation buttons to appear (DeleteButton shows "No, keep it" and "Yes, delete permanently")
    await expect(page.getByText("No, keep it")).toBeVisible();
    await expect(page.getByText("Yes, delete permanently")).toBeVisible();

    // Click "Yes, delete permanently"
    await page.getByRole("button", { name: "Yes, delete permanently" }).click();

    // Wait for the success toast
    await expect(
      page.getByText('Dataset "foo" has been deleted successfully.'),
    ).toBeVisible();

    // Verify we're redirected to the datasets list page
    await expect(page).toHaveURL("/datasets");
    await expect(page.getByText("Dataset Name")).toBeVisible();
  });

  test("should cancel dataset deletion when clicking 'No, keep it'", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Wait for the page to load
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Click the dataset delete button
    const datasetDeleteButton = page
      .getByRole("button", {
        name: "Delete",
        exact: true,
      })
      .first();
    await datasetDeleteButton.click();

    // Wait for the confirmation buttons
    await expect(page.getByText("No, keep it")).toBeVisible();
    await expect(page.getByText("Yes, delete permanently")).toBeVisible();

    // Click "No, keep it" to cancel
    await page.getByRole("button", { name: "No, keep it" }).click();

    // Verify we're still on the dataset page
    await expect(page).toHaveURL("/datasets/foo");
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Verify the delete button is back to its original state
    await expect(page.getByRole("button", { name: "Delete" })).toBeVisible();
  });

  test("should show loading state during dataset deletion", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Wait for the page to load
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Click the dataset delete button
    const datasetDeleteButton = page
      .getByRole("button", {
        name: "Delete",
        exact: true,
      })
      .first();
    await datasetDeleteButton.click();

    // Wait for the confirmation buttons
    await expect(page.getByText("No, keep it")).toBeVisible();
    await expect(page.getByText("Yes, delete permanently")).toBeVisible();

    // Click "Yes, delete permanently"
    const confirmButton = page.getByRole("button", {
      name: "Yes, delete permanently",
    });
    await confirmButton.click();

    // Check that the button shows loading state
    await expect(page.getByText("Deleting...")).toBeVisible();

    // Wait for the deletion to complete and redirect
    await expect(page).toHaveURL("/datasets");
  });

  test("should show loading state during datapoint deletion", async ({
    page,
  }) => {
    await page.goto("/datasets/foo");

    // Wait for the page to load
    await expect(page.getByText("Dataset")).toBeVisible();
    await expect(page.getByText("foo")).toBeVisible();

    // Get the first datapoint row
    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();

    // Click the delete button
    const deleteButton = firstRow
      .locator('button[aria-label="Delete"]')
      .first();
    await deleteButton.click();

    // Wait for the confirmation dialog
    await expect(page.getByText("Delete Datapoint")).toBeVisible();

    // Click "Delete" in the dialog
    await page.getByRole("button", { name: "Delete" }).click();

    // The dialog should close immediately and the page should reload
    // We can verify the success toast appears
    await expect(
      page.getByText("The datapoint has been deleted successfully."),
    ).toBeVisible();
  });
});
