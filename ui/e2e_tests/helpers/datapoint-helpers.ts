import { expect, type Page } from "@playwright/test";
import { v7 as uuidv7 } from "uuid";

/**
 * Options for creating a datapoint from an inference
 */
export interface CreateDatapointFromInferenceOptions {
  /**
   * The inference ID to navigate to (e.g., "0196374b-0d7a-7a22-b2d2-598a14f2eacc")
   */
  inferenceId?: string;
  /**
   * The dataset name to create/use
   * If not provided, a unique name will be generated
   */
  datasetName?: string;
  /**
   * Whether to wait for navigation to the datapoint page after creation
   * Default: true
   */
  waitForNavigation?: boolean;
}

/**
 * Creates a datapoint from an inference by clicking "Add to dataset",
 * creating a new dataset, and selecting "Inference Output".
 *
 * This helper automates the common pattern of:
 * 1. Navigate to an inference page
 * 2. Click "Add to dataset"
 * 3. Fill in dataset name
 * 4. Click create option
 * 5. Click "Inference Output"
 * 6. Wait for toast notification
 * 7. Click "View" in toast
 * 8. Navigate to the new datapoint page
 *
 * @param page - The Playwright Page object
 * @param options - Configuration options for creating the datapoint
 * @returns The dataset name used (useful if auto-generated)
 *
 * @example
 * ```ts
 * const datasetName = await createDatapointFromInference(page, {
 *   inferenceId: "0196374b-0d7a-7a22-b2d2-598a14f2eacc",
 *   datasetName: "my-test-dataset"
 * });
 * ```
 */
export async function createDatapointFromInference(
  page: Page,
  options?: CreateDatapointFromInferenceOptions,
): Promise<string> {
  // Collect an inference if not provided
  const inferenceId =
    options?.inferenceId || "0196c682-72e0-7c83-a92b-9d1a3c7630f2"; // default is a `write_haiku` inference

  // Generate a unique dataset name if not provided
  const datasetName = options?.datasetName || `test_dataset_${uuidv7()}`;

  // Navigate to the inference page
  await page.goto(`/observability/inferences/${inferenceId}`);
  await page.waitForLoadState("networkidle");

  // Click on the "Add to dataset" button
  await page.getByText("Add to dataset").click();

  // Wait for the CommandInput by its placeholder text to be visible
  const commandInput = page.getByPlaceholder("Create or find a dataset");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);

  // Wait for the "Create" option to appear in the dropdown
  const createOption = page
    .locator("[cmdk-item]")
    .filter({ hasText: datasetName });
  await createOption.waitFor({ state: "visible" });
  await createOption.click();

  // Click on the "Inference Output" button
  await page.getByText("Inference Output").click();

  // Wait for the toast to appear with success message
  await expect(
    page
      .getByRole("region", { name: /notifications/i })
      .getByText("New Datapoint"),
  ).toBeVisible();

  // Wait for and click on the "View" button in the toast
  const viewButton = page
    .getByRole("region", { name: /notifications/i })
    .getByText("View");
  await viewButton.waitFor({ state: "visible" });
  await viewButton.click();

  // Wait for navigation to the new page
  if (options?.waitForNavigation !== false) {
    await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
      timeout: 10000,
    });
  }

  return datasetName;
}
