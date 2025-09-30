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
  await page.waitForTimeout(500);

  // Find the CommandInput and create the dataset
  const commandInput = page.getByPlaceholder("Create or find a dataset...");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);
  await page.waitForTimeout(500);

  // Click on the CommandItem to create the dataset
  const createOption = page.locator('div[data-value^="create-"][cmdk-item]');
  await createOption.click();

  // Click on the "Inference Output" button
  await page.getByText("Inference Output").click();

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
    await page.waitForTimeout(500);

    // Select evaluation
    await page.getByText("Select an evaluation").click();
    await page.waitForTimeout(500);
    await page.getByRole("option", { name: "entity_extraction" }).click();
    await page.waitForTimeout(500);

    // Select the dataset we created
    await page.getByText("Select a dataset").click();
    await page.waitForTimeout(500);

    // Type to filter/search for our dataset
    const datasetInput = page.getByPlaceholder("Find a dataset...");
    await datasetInput.fill(datasetName);
    await page.waitForTimeout(500);

    // Click on our dataset
    await page.getByRole("option", { name: datasetName }).click();
    await page.waitForTimeout(500);

    // Select variant
    await page.getByText("Select a variant").click();
    await page.waitForTimeout(500);
    await page
      .getByRole("option", { name: "gpt4o_mini_initial_prompt" })
      .click();
    await page.waitForTimeout(500);

    // Set concurrency and launch the evaluation (which saves to localStorage)
    await page.getByTestId("concurrency-limit").fill("1");
    await page.waitForTimeout(500);

    // Click Launch to submit the form (saves to localStorage via onSubmit)
    await page.getByRole("button", { name: "Launch" }).click();
    await page.waitForTimeout(2000);

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
    await page.waitForTimeout(500);

    // Confirm deletion
    await expect(
      page.getByText(`Are you sure you want to delete the dataset`),
    ).toBeVisible();
    await page.getByRole("button", { name: "Delete" }).click();
    await page.waitForTimeout(500);

    // Wait for the dataset to be deleted
    await expect(datasetRow).not.toBeVisible({ timeout: 10000 });

    // Navigate back to evaluations
    await page.goto("/evaluations");
    await page.waitForLoadState("networkidle");

    // Open the modal again
    await page.getByText("New Run").click();
    await page.waitForTimeout(1000);

    // Verify the evaluation is still selected (from localStorage)
    await expect(page.getByText("entity_extraction")).toBeVisible();

    // Verify the variant is still selected (from localStorage)
    await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

    // Verify the dataset field is cleared (not showing the deleted dataset)
    // The dataset selector should show placeholder text
    const datasetTrigger = page
      .locator('button[role="combobox"]')
      .filter({ hasText: "Select a dataset" });
    await expect(datasetTrigger).toBeVisible();

    // Verify our deleted dataset name is NOT visible in the modal
    await expect(page.getByText(datasetName)).not.toBeVisible();
  });
});
