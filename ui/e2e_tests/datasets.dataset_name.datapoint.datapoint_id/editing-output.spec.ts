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
