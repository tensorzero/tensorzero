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
  const commandInput = page.getByPlaceholder("Create or find a dataset...");
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

test("should be able to add, edit, and delete tags", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0196368f-1ae8-7551-b5df-9a61593eb307",
  );
  await page.waitForLoadState("networkidle");
  const datasetName =
    "test_tags_dataset_" + Math.random().toString(36).substring(2, 15);

  // Create a new datapoint from an inference
  await page.getByText("Add to dataset").click();
  const commandInput = page.getByPlaceholder("Create or find a dataset...");
  await commandInput.waitFor({ state: "visible" });
  await commandInput.fill(datasetName);
  const createOption = page
    .locator("[cmdk-item]")
    .filter({ hasText: datasetName });
  await createOption.waitFor({ state: "visible" });
  await createOption.click();
  await page.getByText("Inference Output").click();
  await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
    timeout: 10000,
  });

  // Enter edit mode
  await page.getByRole("button", { name: "Edit" }).click();

  // Wait for the tags section to be visible
  await expect(page.getByText("Tags")).toBeVisible();

  // Verify the tags table is present but empty initially
  await expect(page.getByText("No tags found")).toBeVisible();

  // Test 1: Add a new tag
  const testKey1 = "environment";
  const testValue1 = "test";
  
  await page.getByPlaceholder("Key").fill(testKey1);
  await page.getByPlaceholder("Value").fill(testValue1);
  await page.getByRole("button", { name: "Add" }).click();

  // Verify the tag appears in the table
  await expect(page.locator("table")).toContainText(testKey1);
  await expect(page.locator("table")).toContainText(testValue1);

  // Test 2: Add another tag to test sorting
  const testKey2 = "author";
  const testValue2 = "e2e-test";
  
  await page.getByPlaceholder("Key").fill(testKey2);
  await page.getByPlaceholder("Value").fill(testValue2);
  await page.getByRole("button", { name: "Add" }).click();

  // Verify both tags appear in the table (should be sorted alphabetically)
  await expect(page.locator("table")).toContainText(testKey2);
  await expect(page.locator("table")).toContainText(testValue2);

  // Test 3: Try to add a system tag (should be prevented)
  const systemKey = "tensorzero::blocked";
  const systemValue = "should_not_work";
  
  await page.getByPlaceholder("Key").fill(systemKey);
  await page.getByPlaceholder("Value").fill(systemValue);
  
  // The Add button should be disabled
  await expect(page.getByRole("button", { name: "Add" })).toBeDisabled();
  
  // Should show error message
  await expect(page.getByText("System tags (starting with \"tensorzero::\") cannot be added manually.")).toBeVisible();

  // Clear the system tag input
  await page.getByPlaceholder("Key").clear();
  await page.getByPlaceholder("Value").clear();

  // Test 4: Save the datapoint and verify tags persist
  await page.getByRole("button", { name: "Save" }).click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Verify tags are still visible in read-only mode
  await expect(page.locator("table")).toContainText(testKey1);
  await expect(page.locator("table")).toContainText(testValue1);
  await expect(page.locator("table")).toContainText(testKey2);
  await expect(page.locator("table")).toContainText(testValue2);

  // Test 5: Edit mode again and delete a tag
  await page.getByRole("button", { name: "Edit" }).click();

  // Find and click the delete button for the first tag (should be "author" due to alphabetical sorting)
  const deleteButton = page.locator("table tr").filter({ hasText: testKey2 }).getByRole("button");
  await deleteButton.click();

  // Verify the tag is removed from the table
  await expect(page.locator("table")).not.toContainText(testKey2);
  await expect(page.locator("table")).not.toContainText(testValue2);

  // But the other tag should still be there
  await expect(page.locator("table")).toContainText(testKey1);
  await expect(page.locator("table")).toContainText(testValue1);

  // Test 6: Save again and reload the page to verify persistence
  await page.getByRole("button", { name: "Save" }).click();
  
  // Reload the page to verify tags persist across page loads
  await page.reload();
  await page.waitForLoadState("networkidle");

  // Verify only the remaining tag is present
  await expect(page.locator("table")).toContainText(testKey1);
  await expect(page.locator("table")).toContainText(testValue1);
  await expect(page.locator("table")).not.toContainText(testKey2);
  await expect(page.locator("table")).not.toContainText(testValue2);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
