import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { createDatapointFromInference } from "../helpers/datapoint-helpers";

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

test("should be able to add, edit and save a json datapoint", async ({
  page,
}) => {
  const datasetName =
    "test_json_dataset_" + Math.random().toString(36).substring(2, 15);

  // Create a new datapoint from an inference
  await createDatapointFromInference(page, {
    inferenceId: "0196368f-1ae8-7551-b5df-9a61593eb307",
    datasetName,
  });

  await expect(page.getByText("Custom")).not.toBeVisible();

  // Click the edit button
  await page.getByRole("button", { name: "Edit" }).click();

  // Edit the input
  const topic = v7();
  const input = `{"topic":"${topic}"}`;
  await page.locator("div[contenteditable='true']").first().fill(input);

  // Edit the output
  const output = `{
    "person": [
      "Ian Thorpe",
      "Garry Kasparov"
    ],
    "organization": [],
    "location": [
      "Australia",
      "Russia"
    ],
    "miscellaneous": [
    ]
  }`;
  await page.locator("div[contenteditable='true']").last().fill(output);

  // Save the datapoint
  await page.getByRole("button", { name: "Save" }).click();

  // Wait for save to complete
  await page.waitForLoadState("networkidle");

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Assert that both input and output are updated
  await expect(page.getByText(input, { exact: true })).toBeVisible();
  await expect(page.getByText("Garry Kasparov")).toBeVisible();

  // Should show "Custom" badge and link original inference
  await expect(page.getByText("Custom", { exact: true })).toBeVisible();
  await expect(page.getByText("Inference", { exact: true })).toBeVisible();
});

test("should be able to add, edit, and delete tags", async ({ page }) => {
  const datasetName =
    "test_tags_dataset_" + Math.random().toString(36).substring(2, 15);

  // Create a new datapoint from an inference
  await createDatapointFromInference(page, {
    inferenceId: "0196a0ea-c165-7b93-85e9-0e9f2ff0fcea",
    datasetName,
  });

  // Enter edit mode
  await page.getByRole("button", { name: "Edit" }).click();

  // Wait for the tags section to be visible
  await expect(page.getByRole("heading", { name: "Tags" })).toBeVisible();

  // Verify the tags section is present (may have system tags or be empty initially)
  // Use a more specific locator that targets the tags section
  const tagsSection = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Tags" }) });
  await expect(tagsSection.locator("table")).toBeVisible();

  // Wait for the input fields to be visible and ready
  await tagsSection.getByPlaceholder("Key").waitFor({ state: "visible" });
  await tagsSection.getByPlaceholder("Value").waitFor({ state: "visible" });

  // Test 1: Add a new tag
  const testKey1 = "environment";
  const testValue1 = "test";

  await tagsSection.getByPlaceholder("Key").fill(testKey1);
  await tagsSection.getByPlaceholder("Value").fill(testValue1);
  await page.getByRole("button", { name: "Add" }).click();

  // Wait for inputs to be cleared after adding (indicates operation completed)
  await expect(tagsSection.getByPlaceholder("Key")).toHaveValue("");

  // Wait for the tag to appear in the table before proceeding
  await expect(tagsSection.locator("table")).toContainText(testKey1);
  await expect(tagsSection.locator("table")).toContainText(testValue1);

  // Test 2: Add another tag to test sorting
  const testKey2 = "author";
  const testValue2 = "e2e-test";

  await tagsSection.getByPlaceholder("Key").fill(testKey2);
  await tagsSection.getByPlaceholder("Value").fill(testValue2);
  await page.getByRole("button", { name: "Add" }).click();

  // Wait for inputs to be cleared after adding (indicates operation completed)
  await expect(tagsSection.getByPlaceholder("Key")).toHaveValue("");

  // Wait for the tag to appear in the table before proceeding
  await expect(tagsSection.locator("table")).toContainText(testKey2);
  await expect(tagsSection.locator("table")).toContainText(testValue2);

  // Test 3: Try to add a system tag (should be prevented)
  const systemKey = "tensorzero::blocked";
  const systemValue = "should_not_work";

  await tagsSection.getByPlaceholder("Key").fill(systemKey);
  await tagsSection.getByPlaceholder("Value").fill(systemValue);

  // Wait for the button state to update based on input validation
  await page.waitForTimeout(100);

  // The Add button should be disabled
  await expect(page.getByRole("button", { name: "Add" })).toBeDisabled();

  // Should show error message
  await expect(
    page.getByText(
      'System tags (starting with "tensorzero::") cannot be added manually.',
    ),
  ).toBeVisible();

  // Clear the system tag input
  await tagsSection.getByPlaceholder("Key").clear();
  await tagsSection.getByPlaceholder("Value").clear();

  // Test 4: Save the datapoint and verify tags persist
  await page.getByRole("button", { name: "Save" }).click();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Verify tags are still visible in read-only mode
  const tagsSection2 = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Tags" }) });
  await expect(tagsSection2.locator("table")).toContainText(testKey1);
  await expect(tagsSection2.locator("table")).toContainText(testValue1);
  await expect(tagsSection2.locator("table")).toContainText(testKey2);
  await expect(tagsSection2.locator("table")).toContainText(testValue2);

  // Test 5: Edit mode again and delete a tag
  await page.getByRole("button", { name: "Edit" }).click();

  // Find and click the delete button for the first tag (should be "author" due to alphabetical sorting)
  const tagsSection3 = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Tags" }) });
  const deleteButton = tagsSection3
    .locator("table tr")
    .filter({ hasText: testKey2 })
    .getByRole("button");
  await deleteButton.click();

  // Verify the tag is removed from the table
  await expect(tagsSection3.locator("table")).not.toContainText(testKey2);
  await expect(tagsSection3.locator("table")).not.toContainText(testValue2);

  // But the other tag should still be there
  await expect(tagsSection3.locator("table")).toContainText(testKey1);
  await expect(tagsSection3.locator("table")).toContainText(testValue1);

  // Test 6: Edit an existing tag by overwriting it
  const newTestValue1 = "production"; // New value for environment tag

  await tagsSection3.getByPlaceholder("Key").fill(testKey1); // Use same key "environment"
  await tagsSection3.getByPlaceholder("Value").fill(newTestValue1);
  await page.getByRole("button", { name: "Add" }).click();

  // Wait for inputs to be cleared after adding (indicates operation completed)
  await expect(tagsSection3.getByPlaceholder("Key")).toHaveValue("");

  // Wait for the tag to be updated in the table
  await expect(tagsSection3.locator("table")).toContainText(testKey1);
  await expect(tagsSection3.locator("table")).toContainText(newTestValue1);
  await expect(tagsSection3.locator("table")).not.toContainText(testValue1); // Old value should be gone

  // Test 7: Save again and verify persistence after redirect
  await page.getByRole("button", { name: "Save" }).click();

  // Wait for the save to complete and potential redirect (new datapoint ID)
  await page.waitForTimeout(100);
  await page.waitForLoadState("networkidle");

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();

  // Verify we're still on a datapoint page (URL should have changed to new ID)
  await expect(page.url()).toMatch(
    new RegExp(`/datasets/${datasetName}/datapoint/[^/]+$`),
  );

  // Verify only the remaining tag with updated value is present after the save/redirect
  const tagsSection4 = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Tags" }) });
  await expect(tagsSection4.locator("table")).toContainText(testKey1);
  await expect(tagsSection4.locator("table")).toContainText(newTestValue1); // New value
  await expect(tagsSection4.locator("table")).not.toContainText(testValue1); // Old value should be gone
  await expect(tagsSection4.locator("table")).not.toContainText(testKey2);
  await expect(tagsSection4.locator("table")).not.toContainText(testValue2);

  // Test 8: Reload the page to verify persistence
  await page.reload();
  await page.waitForLoadState("networkidle");

  // Verify tags are still present after page reload with the updated value
  const tagsSection5 = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Tags" }) });
  await expect(tagsSection5.locator("table")).toContainText(testKey1);
  await expect(tagsSection5.locator("table")).toContainText(newTestValue1); // New value
  await expect(tagsSection5.locator("table")).not.toContainText(testValue1); // Old value should be gone
  await expect(tagsSection5.locator("table")).not.toContainText(testKey2);
  await expect(tagsSection5.locator("table")).not.toContainText(testValue2);
});

test("should be able to rename a datapoint", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );
  await page.waitForLoadState("networkidle");
  await expect(
    page.getByRole("button", { name: "Rename datapoint" }),
  ).toBeVisible();

  // Click on the Add to dataset button
  await page.getByRole("button", { name: "Rename datapoint" }).click();

  // Wait for the datapoint name input by its label to be visible
  const datapointNameInput = page.getByLabel("Datapoint name");
  await datapointNameInput.waitFor({ state: "visible" });
  await datapointNameInput.fill("New Datapoint Name");

  // Click on the Save button
  await page.getByRole("button", { name: "Save" }).click();

  // Datapoint name should be updated
  await expect(
    page.getByText("New Datapoint Name", { exact: true }),
  ).toBeVisible();
});

test("should be able to cancel renaming a datapoint", async ({ page }) => {
  await page.goto(
    "/datasets/foo/datapoint/0196374b-d575-77b3-ac22-91806c67745c",
  );
  await page.waitForLoadState("networkidle");
  await expect(
    page.getByRole("button", { name: "Rename datapoint" }),
  ).toBeVisible();

  // Click on the Add to dataset button
  await page.getByRole("button", { name: "Rename datapoint" }).click();

  // Wait for the datapoint name input by its label to be visible
  const datapointNameInput = page.getByLabel("Datapoint name");
  await datapointNameInput.waitFor({ state: "visible" });
  await datapointNameInput.fill("Renamed Datapoint Name");

  // Click on the Cancel button
  await page.getByRole("button", { name: "Cancel" }).click();

  // We should not have updated the datapoint name
  await expect(
    page.getByText("Renamed Datapoint Name", { exact: false }),
  ).not.toBeVisible();
});
