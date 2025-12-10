import { test, expect } from "@playwright/test";
import { v7 as uuidv7 } from "uuid";
import { createDatapointFromInference } from "./helpers/datapoint-helpers";

test.describe("Clone Datapoint", () => {
  test("should clone a datapoint to a new dataset", async ({ page }) => {
    // Step 1: Create an initial datapoint from an inference
    const sourceDatasetName = await createDatapointFromInference(page);

    // Get the source datapoint URL and ID
    const sourceUrl = page.url();
    expect(sourceUrl).toContain(sourceDatasetName);
    const sourceDatapointId = sourceUrl.split("/datapoint/")[1]?.split("?")[0];
    expect(sourceDatapointId).toBeDefined();

    // Step 2: Clone to a new dataset
    const targetDatasetName = `clone_target_${uuidv7()}`;

    // Click on the Clone button (Dataset selector)
    await page.getByText("Clone").click();

    // Wait for the CommandInput to be visible
    const commandInput = page.getByPlaceholder("Create or find a dataset");
    await commandInput.waitFor({ state: "visible" });
    await commandInput.fill(targetDatasetName);

    // Wait for the "Create" option to appear and click it
    const createOption = page
      .locator("[cmdk-item]")
      .filter({ hasText: targetDatasetName });
    await createOption.waitFor({ state: "visible" });
    await createOption.click();

    // Step 3: Wait for success toast
    await expect(
      page
        .getByRole("region", { name: /notifications/i })
        .getByText("Datapoint Cloned"),
    ).toBeVisible({ timeout: 10000 });

    // Step 4: Navigate to the target dataset page to verify the datapoint was created
    await page.goto(`/datasets/${targetDatasetName}`);
    await page.waitForLoadState("networkidle");

    // Verify the dataset page loads and has at least one datapoint
    await expect(page.getByRole("heading", { name: "Dataset" })).toBeVisible({
      timeout: 10000,
    });

    // Verify there's at least one datapoint in the table
    const rows = page.locator("tbody tr");
    await expect(rows).toHaveCount(1, { timeout: 10000 });
  });

  test("should clone a datapoint to an existing dataset", async ({ page }) => {
    // Navigate away and create the target dataset
    const targetDatasetName = `existing_target_${uuidv7()}`;
    await createDatapointFromInference(page, {
      datasetName: targetDatasetName,
    });

    // Verify we're on a datapoint page (label text, not a heading)
    await expect(page.locator("text=Datapoint").first()).toBeVisible({
      timeout: 10000,
    });

    // Step 2: Clone to the existing target dataset
    await page.getByText("Clone").click();

    const commandInput = page.getByPlaceholder("Create or find a dataset");
    await commandInput.waitFor({ state: "visible" });
    await commandInput.fill(targetDatasetName);

    // Select the existing dataset
    const existingDatasetOption = page
      .locator("[cmdk-item]")
      .filter({ hasText: targetDatasetName });
    await existingDatasetOption.waitFor({ state: "visible" });
    await existingDatasetOption.click();

    // Step 3: Wait for success toast
    await expect(
      page
        .getByRole("region", { name: /notifications/i })
        .getByText("Datapoint Cloned"),
    ).toBeVisible({ timeout: 10000 });

    // Step 4: Navigate to the target dataset page to verify the datapoint was cloned
    await page.goto(`/datasets/${targetDatasetName}`);
    await page.waitForLoadState("networkidle");

    // Verify there are now 2 datapoints (original + clone)
    const rows = page.locator("tbody tr");
    await expect(rows).toHaveCount(2, { timeout: 10000 });
  });

  test("should preserve source_inference_id when cloning", async ({ page }) => {
    // Step 1: Create a datapoint from an inference (this sets source_inference_id)
    await createDatapointFromInference(page);

    // Get the source inference ID by checking the page content
    // The datapoint page should show "Source Inference" link - use .first() since there may be multiple inference links
    const sourceInferenceLink = page
      .locator('a[href*="/observability/inferences/"]')
      .first();
    await expect(sourceInferenceLink).toBeVisible({ timeout: 10000 });
    const sourceInferenceHref = await sourceInferenceLink.getAttribute("href");
    const sourceInferenceId = sourceInferenceHref?.split("/inferences/")[1];
    expect(sourceInferenceId).toBeDefined();

    // Step 2: Clone to a new dataset
    const targetDatasetName = `clone_source_test_${uuidv7()}`;
    await page.getByText("Clone").click();

    const commandInput = page.getByPlaceholder("Create or find a dataset");
    await commandInput.waitFor({ state: "visible" });
    await commandInput.fill(targetDatasetName);

    const createOption = page
      .locator("[cmdk-item]")
      .filter({ hasText: targetDatasetName });
    await createOption.waitFor({ state: "visible" });
    await createOption.click();

    // Wait for success toast
    await expect(
      page
        .getByRole("region", { name: /notifications/i })
        .getByText("Datapoint Cloned"),
    ).toBeVisible({ timeout: 10000 });

    // Step 3: Navigate to the cloned datapoint and verify source_inference_id
    await page.goto(`/datasets/${targetDatasetName}`);
    await page.waitForLoadState("networkidle");

    // Click on the datapoint ID link in the first cell to navigate to it
    const firstRow = page.locator("tbody tr").first();
    const idLink = firstRow.locator("td").first().locator("a");
    await idLink.click();
    await page.waitForLoadState("networkidle");

    // Verify the cloned datapoint also shows the same Source Inference link
    const clonedInferenceLink = page
      .locator('a[href*="/observability/inferences/"]')
      .first();
    await expect(clonedInferenceLink).toBeVisible({ timeout: 10000 });
    const clonedInferenceHref = await clonedInferenceLink.getAttribute("href");
    const clonedInferenceId = clonedInferenceHref?.split("/inferences/")[1];

    expect(clonedInferenceId).toBe(sourceInferenceId);
  });

  test("should not show clone button when in editing mode", async ({
    page,
  }) => {
    // Create a datapoint
    await createDatapointFromInference(page);

    // Verify clone button is visible initially
    await expect(page.getByText("Clone")).toBeVisible();

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Verify clone button is no longer visible
    await expect(page.getByText("Clone")).not.toBeVisible();

    // Cancel editing
    await page.getByRole("button", { name: "Cancel" }).click();

    // Verify clone button is visible again
    await expect(page.getByText("Clone")).toBeVisible();
  });
});
