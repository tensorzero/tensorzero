import { test, expect, type Page } from "@playwright/test";
import { randomUUID } from "crypto";

// Helper function to create a unique dataset for testing
async function createTestDataset(
  page: Page,
  baseName: string = "test-eval-dataset",
): Promise<string> {
  const datasetName = `${baseName}-${Date.now()}-${randomUUID()}`;

  await page.goto(
    "/observability/inferences/0196368f-1ae8-7551-b5df-9a61593eb307",
  );
  await page.waitForLoadState("networkidle");

  // Click on the Add to dataset button
  await page.getByText("Add to dataset").click();

  // Find the CommandInput and create the dataset
  const commandInput = page.getByPlaceholder("Create or find a dataset...");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);

  // Click on the CommandItem to create the dataset
  const createOption = page.locator('div[data-value^="create-"][cmdk-item]');
  await expect(createOption).toBeVisible();
  await createOption.click();

  // Click on the "Inference Output" button
  const inferenceOutputButton = page.getByText("Inference Output");
  await expect(inferenceOutputButton).toBeVisible();
  await inferenceOutputButton.click();

  // Wait for the toast to appear with success message
  await expect(
    page
      .getByRole("region", { name: /notifications/i })
      .getByText("New Datapoint"),
  ).toBeVisible();

  return datasetName;
}

test.describe("Launch Evaluation Modal - Deleted Dataset", () => {
  test("should clear deleted dataset from localStorage when modal reopens", async ({
    page,
  }) => {
    // Create a unique test dataset
    const datasetName = await createTestDataset(page, "eval-localstorage-test");

    // Navigate to evaluations page
    await page.goto("/evaluations");
    await page.waitForLoadState("networkidle");
    await expect(page.getByText("New Run")).toBeVisible();

    // Open the launch evaluation modal
    await page.getByText("New Run").click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByText("Select an evaluation")).toBeVisible();

    // Select evaluation
    await page.getByText("Select an evaluation").click();
    const evaluationOption = page.getByRole("option", {
      name: "entity_extraction",
    });
    await expect(evaluationOption).toBeVisible();
    await evaluationOption.click();
    await expect(page.getByText("Select a dataset")).toBeVisible();

    // Select the dataset we created
    await page.getByText("Select a dataset").click();

    // Type to filter/search for our dataset
    const datasetInput = page.getByPlaceholder("Find a dataset...");
    await datasetInput.fill(datasetName);
    const datasetOption = page.locator(`[data-dataset-name="${datasetName}"]`);
    await expect(datasetOption).toBeVisible();

    // Click on our dataset
    await datasetOption.click();
    await expect(page.getByText("Select a variant")).toBeVisible();

    // Select variant
    await page.getByText("Select a variant").click();
    const variantOption = page.getByRole("option", {
      name: "gpt4o_mini_initial_prompt",
    });
    await expect(variantOption).toBeVisible();
    await variantOption.click();

    // Set concurrency and launch the evaluation (which saves to localStorage)
    const concurrencyInput = page.getByTestId("concurrency-limit");
    await expect(concurrencyInput).toBeVisible();
    await concurrencyInput.fill("1");

    // Click Launch to submit the form (saves to localStorage via onSubmit)
    const launchButton = page.getByRole("button", { name: "Launch" });
    await expect(launchButton).toBeEnabled();
    await launchButton.click();

    // Wait for the dialog to close after launching
    await expect(page.getByRole("dialog")).toBeHidden();

    // The evaluation will likely fail because the dataset has wrong function datapoints
    // But the form was submitted and saved to localStorage, which is what matters

    // Now delete the dataset
    await page.goto("/datasets");
    await page.waitForLoadState("networkidle");

    // Find and click delete button for our dataset
    const datasetRow = page.locator(`tr#${datasetName}`);
    await expect(datasetRow).toBeVisible();

    const deleteButton = datasetRow.locator("button").last();
    await deleteButton.click();

    // Confirm deletion
    await expect(
      page.getByText(`Are you sure you want to delete the dataset`),
    ).toBeVisible();
    await page.getByRole("button", { name: "Delete" }).click();

    // Wait for the dataset to be deleted
    await expect(datasetRow).not.toBeVisible({ timeout: 10000 });

    // Navigate back to evaluations
    await page.goto("/evaluations");
    await page.waitForLoadState("networkidle");

    // Open the modal again
    await page.getByText("New Run").click();

    // Wait for the dialog to be visible
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();

    // Verify the evaluation is still selected (from localStorage)
    const evaluationCombobox = dialog
      .getByRole("combobox")
      .filter({ hasText: "entity_extraction" });
    await expect(evaluationCombobox).toBeVisible();

    // Verify the variant is still selected (from localStorage)
    const variantCombobox = dialog
      .getByRole("combobox")
      .filter({ hasText: "gpt4o_mini_initial_prompt" });
    await expect(variantCombobox).toBeVisible();

    // Verify the dataset field is cleared (not showing the deleted dataset)
    // The dataset selector should show placeholder text
    const datasetTrigger = dialog
      .locator('button[role="combobox"]')
      .filter({ hasText: "Select a dataset" });
    await expect(datasetTrigger).toBeVisible();

    // Verify our deleted dataset name is NOT visible in the modal
    await expect(dialog.getByText(datasetName)).not.toBeVisible();
  });
});
