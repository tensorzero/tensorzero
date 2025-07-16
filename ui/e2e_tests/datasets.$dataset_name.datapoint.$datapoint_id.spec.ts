import { test, expect } from "@playwright/test";
import { v7 } from "uuid";

test("should show the datapoint detail page", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );
  await expect(page.getByText("Input")).toBeVisible();

  // Clicking episode ID opens episode page
  await page
    .getByRole("link", { name: "0193da94-231b-72e0-bda1-dfd0f681462d" })
    .click();
  await expect(page).toHaveURL(
    /\/observability\/episodes\/0193da94-231b-72e0-bda1-dfd0f681462d/,
  );
  await page.goBack();

  // Clicking inference ID opens inference page
  await page
    .getByRole("link", { name: "019480f9-d420-73b1-9619-81d71adc18a5" })
    .click();
  await expect(page).toHaveURL(
    /\/observability\/inferences\/019480f9-d420-73b1-9619-81d71adc18a5/,
  );
  await page.goBack();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to add, edit and save a datapoint", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0196368f-1ae8-7551-b5df-9a61593eb307",
  );
  await page.waitForLoadState("networkidle");
  const datasetName =
    "test_json_dataset_" + Math.random().toString(36).substring(2, 15);

  // Click on the Add to dataset button
  await page.getByText("Add to dataset").click();

  // Wait for the CommandInput by its placeholder text to be visible
  const commandInput = page.getByPlaceholder("Create or find dataset...");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);

  // Wait for the "Create" option to appear in the dropdown
  const createOption = page
    .locator("[cmdk-item]")
    .filter({ hasText: "Create" });
  await createOption.waitFor({ state: "visible" });
  await createOption.click();

  // Click on the "Inference Output" button
  await page.getByText("Inference Output").click();

  // Wait for navigation to the new page
  await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
    timeout: 10000,
  });

  await expect(page.getByText("Custom")).not.toBeVisible();

  // Click the edit button
  await page.getByRole("button", { name: "Edit" }).click();

  // Edit the input
  const topic = v7();
  const input = `{"topic":"${topic}"}`;

  await page.locator("div[contenteditable='true']").first().fill(input);

  // Save the datapoint
  await page.getByRole("button", { name: "Save" }).click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that the input is updated
  await expect(page.getByText(input)).toBeVisible();

  // Should show "Custom" badge and link original inference
  await expect(page.getByText("Custom")).toBeVisible();
  await expect(page.getByText("Inference", { exact: true })).toBeVisible();
});
