import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { createDatapointFromInference } from "../helpers/datapoint-helpers";
import { saveAndWaitForRedirect, expandShowMoreIfPresent } from "./helpers";

// ============================================================================
// Output Content Block Tests
// ============================================================================

test.describe("Output - Text Blocks", () => {
  test("should add, edit, and delete text block in output", async ({
    page,
  }) => {
    // Create datapoint from inference with chat output
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons
    await expandShowMoreIfPresent(page);

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Add text block to output
    const addTextButton = outputSection
      .getByRole("button", { name: "Text" })
      .last();
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    const textEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor.waitFor({ state: "visible" });

    const textContent = v7();
    await textEditor.fill(`Output text: ${textContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify text block was added
    await expect(page.getByText(textContent)).toBeVisible();

    // Step 2: Edit the text block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const textEditor2 = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor2.waitFor({ state: "visible" });

    const textContent2 = v7();
    await textEditor2.fill(`Edited output text: ${textContent2}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify edited content is visible and old content is gone
    await expect(page.getByText(textContent2)).toBeVisible();
    await expect(page.getByText(textContent)).not.toBeVisible();

    // Step 3: Delete the text block
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = outputSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify text block was deleted
    await expect(page.getByText(textContent2)).not.toBeVisible();
    await expect(page.getByText(textContent)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(textContent2)).not.toBeVisible();
  });
});

test.describe("Output - Tool Call Blocks", () => {
  test("should add, edit, and delete tool call block in output", async ({
    page,
  }) => {
    // Create datapoint from inference with tool-enabled function
    await createDatapointFromInference(page, {
      inferenceId: "0196a0e5-ba06-7fd1-bf50-aed8fc9cf2ae",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons
    await expandShowMoreIfPresent(page);

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Add tool call to output
    const addToolCallButton = outputSection
      .getByRole("button", { name: "Tool Call" })
      .last();
    await expect(addToolCallButton).toBeVisible();
    await addToolCallButton.click();

    // Expand all "Show more" buttons again (tool call might be collapsed)
    await expandShowMoreIfPresent(page);

    // Fill in tool call details
    const toolId = "tool_" + v7();
    const toolName = "think"; // Use valid tool name for this function

    // Find the tool call inputs using data-testid (use .last() since there may be existing tool calls)
    const nameInput = outputSection.getByTestId("tool-name-input").last();
    await nameInput.waitFor({ state: "visible" });
    await nameInput.fill(toolName);

    const idInput = outputSection.getByTestId("tool-id-input").last();
    await idInput.fill(toolId);

    const argsEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditor.fill('{"thought": "test thought"}');

    // Save
    await saveAndWaitForRedirect(page);

    // Verify tool call visible (check for unique content and ID)
    await expect(page.getByText("test thought")).toBeVisible();
    // TODO (#4058): we are not roundtripping IDs
    // await expect(page.getByText(toolId)).toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await expect(page.getByText("test thought")).toBeVisible();
    // TODO (#4058): we are not roundtripping IDs
    // await expect(page.getByText(toolId)).toBeVisible();

    // Edit again and verify ID is preserved
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    // Modify the arguments
    const argsEditorEdit = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditorEdit.waitFor({ state: "visible" });
    await argsEditorEdit.fill('{"thought": "updated thought"}');

    // Save
    await saveAndWaitForRedirect(page);

    // TODO (#4058): we are not roundtripping IDs
    // await expect(page.getByText(toolId)).toBeVisible();
    await expect(page.getByText("updated thought")).toBeVisible();

    // Delete tool call
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = outputSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify tool call removed
    await expect(page.getByText(toolId)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(toolId)).not.toBeVisible();
  });
});

test.describe("Output - Thought Blocks", () => {
  test("should add, edit, and delete thought block in output", async ({
    page,
  }) => {
    // Create datapoint from inference with chat output
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons
    await expandShowMoreIfPresent(page);

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Add thought block to output
    const addThoughtButton = outputSection
      .getByRole("button", { name: "Thought" })
      .last();
    await expect(addThoughtButton).toBeVisible();
    await addThoughtButton.click();

    const thoughtEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor.waitFor({ state: "visible" });

    const thoughtContent = v7();
    await thoughtEditor.fill(`Output thought: ${thoughtContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify thought block was added
    await expect(page.getByText(thoughtContent)).toBeVisible();

    // Step 2: Edit the thought block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const thoughtEditor2 = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor2.waitFor({ state: "visible" });

    const thoughtContent2 = v7();
    await thoughtEditor2.fill(`Edited output thought: ${thoughtContent2}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify edited content is visible and old content is gone
    await expect(page.getByText(thoughtContent2)).toBeVisible();
    await expect(page.getByText(thoughtContent)).not.toBeVisible();

    // Step 3: Delete the thought block
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = outputSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify thought block was deleted
    await expect(page.getByText(thoughtContent2)).not.toBeVisible();
    await expect(page.getByText(thoughtContent)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(thoughtContent2)).not.toBeVisible();
  });
});

// ============================================================================
// Output Full Deletion and Re-adding Tests
// ============================================================================

test.describe("Output - Full Deletion", () => {
  test("should delete entire output and add it back for chat datapoint", async ({
    page,
  }) => {
    // Create datapoint from inference with chat output
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons
    await expandShowMoreIfPresent(page);

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Delete entire output
    const deleteOutputButton = outputSection.getByRole("button", {
      name: "Delete output",
    });
    await expect(deleteOutputButton).toBeVisible();
    await deleteOutputButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify "No output" message is visible
    await expect(outputSection.getByText("No output")).toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(outputSection.getByText("No output")).toBeVisible();

    // Enter edit mode again
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Add output back
    const addOutputButton = outputSection.getByRole("button", {
      name: "Output",
    });
    await expect(addOutputButton).toBeVisible();
    await addOutputButton.click();

    // Now we should have content block buttons available - add a text block
    const addTextButton = outputSection
      .getByRole("button", { name: "Text" })
      .last();
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    const textEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor.waitFor({ state: "visible" });

    const textContent = v7();
    await textEditor.fill(`Restored output: ${textContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify content is visible
    await expect(page.getByText(textContent)).toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(textContent)).toBeVisible();
  });

  test("should delete entire output and add it back for JSON datapoint", async ({
    page,
  }) => {
    // Create datapoint from JSON inference (extract_entities function)
    await createDatapointFromInference(page, {
      inferenceId: "0196368f-1ae8-7551-b5df-9a61593eb307",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Delete entire output
    const deleteOutputButton = outputSection.getByRole("button", {
      name: "Delete output",
    });
    await expect(deleteOutputButton).toBeVisible();
    await deleteOutputButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify "No output" message is visible
    await expect(outputSection.getByText("No output")).toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(outputSection.getByText("No output")).toBeVisible();

    // Enter edit mode again
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Add output back (JSON output uses "Output" button)
    const addOutputButton = outputSection.getByRole("button", {
      name: "Output",
    });
    await expect(addOutputButton).toBeVisible();
    await addOutputButton.click();

    // Fill in JSON content
    const jsonEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await jsonEditor.waitFor({ state: "visible" });

    const uniqueValue = v7();
    const jsonContent = `{"person": ["Test_${uniqueValue}"]}`;
    await jsonEditor.fill(jsonContent);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify content is visible (check for unique value)
    await expect(page.getByText(uniqueValue)).toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(uniqueValue)).toBeVisible();
  });
});

// ============================================================================
// JSON Output Editing Tests
// ============================================================================

test.describe("JSON Output - Full Editing", () => {
  test("should add, edit, and delete JSON output", async ({ page }) => {
    // Create datapoint from JSON inference (extract_entities function)
    await createDatapointFromInference(page, {
      inferenceId: "0196368f-1ae8-7551-b5df-9a61593eb307",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Find Output section
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Step 1: Edit existing JSON output
    const jsonEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await jsonEditor.waitFor({ state: "visible" });

    const uniqueValue1 = v7();
    const jsonContent1 = `{"person": ["Person_${uniqueValue1}"], "organization": [], "location": [], "miscellaneous": []}`;
    await jsonEditor.fill(jsonContent1);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify content is visible
    await expect(page.getByText(uniqueValue1)).toBeVisible();

    // Step 2: Edit the JSON content again
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const jsonEditor2 = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await jsonEditor2.waitFor({ state: "visible" });

    const uniqueValue2 = v7();
    const jsonContent2 = `{"person": ["Updated_${uniqueValue2}"], "organization": ["TestOrg"], "location": [], "miscellaneous": []}`;
    await jsonEditor2.fill(jsonContent2);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify new content is visible and old content is gone
    await expect(page.getByText(uniqueValue2)).toBeVisible();
    await expect(page.getByText(uniqueValue1)).not.toBeVisible();

    // Step 3: Delete entire output
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteOutputButton = outputSection.getByRole("button", {
      name: "Delete output",
    });
    await expect(deleteOutputButton).toBeVisible();
    await deleteOutputButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify "No output" message is visible
    await expect(outputSection.getByText("No output")).toBeVisible();
    await expect(page.getByText(uniqueValue2)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(outputSection.getByText("No output")).toBeVisible();
    await expect(page.getByText(uniqueValue2)).not.toBeVisible();
  });
});
