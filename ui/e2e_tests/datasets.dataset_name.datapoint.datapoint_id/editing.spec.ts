import { test, expect } from "@playwright/test";
import { v7 } from "uuid";
import { createDatapointFromInference } from "../helpers/datapoint-helpers";
import { saveAndWaitForRedirect, expandShowMoreIfPresent } from "./helpers";

/**
 * Comprehensive tests for dataset datapoint editing functionality
 * Tests all content block types and editing operations for input messages
 */

// ============================================================================
// System Message Tests
// ============================================================================

test.describe("System Message - Text", () => {
  test("should add, edit, and delete system text", async ({ page }) => {
    // Create a new datapoint from an inference without a system message
    await createDatapointFromInference(page, {
      inferenceId: "0196374b-0d7a-7a22-b2d2-598a14f2eacc",
    });

    // Step 1: Add system text
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const systemSection = page.getByTestId("message-system");
    const addTextButton = systemSection.getByRole("button", { name: "Text" });
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    const systemMessageEditor = page
      .locator("div[contenteditable='true']")
      .first();
    await systemMessageEditor.waitFor({ state: "visible" });

    const systemMessageText1 = v7();
    await systemMessageEditor.fill(
      `You are a helpful assistant. Context: ${systemMessageText1}`,
    );

    // Save
    await saveAndWaitForRedirect(page);

    // Verify system message was added
    await expect(page.getByText(systemMessageText1)).toBeVisible();

    // Step 2: Edit system text
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const systemMessageEditor2 = page
      .locator("div[contenteditable='true']")
      .first();
    await systemMessageEditor2.waitFor({ state: "visible" });

    const systemMessageText2 = v7();
    await systemMessageEditor2.fill(
      `Edited system message: ${systemMessageText2}`,
    );

    // Save
    await saveAndWaitForRedirect(page);

    // Verify system message was edited
    await expect(page.getByText(systemMessageText2)).toBeVisible();
    await expect(page.getByText(systemMessageText1)).not.toBeVisible();

    // Step 3: Delete system text
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteSystemButton = page.getByRole("button", {
      name: "Delete system",
    });
    await expect(deleteSystemButton).toBeVisible();
    await deleteSystemButton.click();

    // Verify "+ Text" button is visible after deletion
    const systemSectionAfterDelete = page
      .locator("div")
      .filter({ hasText: /^system$/ })
      .locator("..");
    await expect(
      systemSectionAfterDelete.getByRole("button", { name: "Text" }),
    ).toBeVisible();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify system message was deleted
    await expect(page.getByText("system", { exact: true })).not.toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText("system", { exact: true })).not.toBeVisible();
  });
});

test.describe("System Message - Template", () => {
  test("should add, delete, re-add, and edit system template", async ({
    page,
  }) => {
    // Use answer_question function which has system_schema
    // This allows us to add a system template (JSON content)
    // Create datapoint
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Step 1: Delete existing system template and re-add in same edit session
    // The inference already has a system message, so we start by deleting it
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteSystemButton = page.getByRole("button", {
      name: "Delete system",
    });
    await expect(deleteSystemButton).toBeVisible();
    await deleteSystemButton.click();

    // Verify visually that "+ Template" button appears after deletion (still in edit mode)
    const systemSection = page.getByTestId("message-system");

    const addTemplateButton = systemSection.getByRole("button", {
      name: "Template",
    });
    await expect(addTemplateButton).toBeVisible();

    // Step 2: Re-add system template (still in same edit session)
    await addTemplateButton.click();

    let templateEditor = page.locator("div[contenteditable='true']").first();
    await templateEditor.waitFor({ state: "visible" });

    const templateValue1 = v7();
    const templateJson1 = JSON.stringify({ secret: templateValue1 }, null, 2);
    await templateEditor.fill(templateJson1);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify template was added
    const templateSection = page
      .getByText("Template:")
      .first()
      .locator("..")
      .locator("..")
      .locator("..")
      .locator("..");
    await expect(templateSection).toBeVisible();
    let codeContent = await templateSection
      .locator(".cm-content")
      .textContent();
    expect(codeContent).toContain(templateValue1);

    // Step 3: Edit the template content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    templateEditor = page.locator("div[contenteditable='true']").first();
    await templateEditor.waitFor({ state: "visible" });

    const templateValue2 = v7();
    const templateJson2 = JSON.stringify({ secret: templateValue2 }, null, 2);
    await templateEditor.fill(templateJson2);

    await saveAndWaitForRedirect(page);

    // Verify template was edited
    await expect(templateSection).toBeVisible();
    codeContent = await templateSection.locator(".cm-content").textContent();
    expect(codeContent).toContain(templateValue2);
    expect(codeContent).not.toContain(templateValue1);

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    codeContent = await templateSection.locator(".cm-content").textContent();
    expect(codeContent).toContain(templateValue2);
  });
});

// ============================================================================
// User Message Content Block Tests
// ============================================================================

test.describe("User Message - Text Blocks", () => {
  test("should add, edit, and delete text block in user message", async ({
    page,
  }) => {
    // Create datapoint from inference without user_schema (answer_question function)
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate by waiting for Save button
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Find a user message section and add text block
    const userSection = page.getByTestId("message-user").first();

    const addTextButton = userSection
      .getByRole("button", { name: "Text" })
      .last();
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    // Find the newly added text editor (last contenteditable in user section)
    const textEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor.waitFor({ state: "visible" });

    // Fill with content
    const textContent = v7();
    await textEditor.fill(`User text content: ${textContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify text block was added
    await expect(page.getByText(textContent)).toBeVisible();

    // Step 2: Edit the text block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    // Find the text editor again
    const textEditor2 = userSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor2.waitFor({ state: "visible" });

    const textContent2 = v7();
    await textEditor2.fill(`Edited user text: ${textContent2}`);

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

    // Find delete button for the newly added content block (last one)
    const deleteButton = userSection
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

test.describe("User Message - Tool Call Blocks", () => {
  test("should add, edit, and delete tool call block in user message", async ({
    page,
  }) => {
    // Create datapoint from inference without user_schema
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Add tool call to user message
    const userSection = page.getByTestId("message-user").first();

    const addToolCallButton = userSection
      .getByRole("button", {
        name: "Tool Call",
      })
      .last();
    await expect(addToolCallButton).toBeVisible();
    await addToolCallButton.click();

    // Fill in tool call details
    const toolId = "tool_" + v7();
    const toolName = "test_tool_" + Math.random().toString(36).substring(2, 10);

    // Find the tool call inputs (should have Name, ID, Arguments fields)
    const nameInput = userSection
      .getByPlaceholder("tool_name")
      .or(userSection.locator('input[value=""]'))
      .first();
    await nameInput.fill(toolName);

    const idInput = userSection.locator("input").nth(1);
    await idInput.fill(toolId);

    const argsEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditor.fill('{"param": "value"}');

    // Save
    await saveAndWaitForRedirect(page);

    // Verify tool call visible (check for tool name and ID)
    await expect(page.getByText(toolName)).toBeVisible();
    await expect(page.getByText(toolId)).toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await expect(page.getByText(toolName)).toBeVisible();
    await expect(page.getByText(toolId)).toBeVisible();

    // Edit again and verify ID is preserved
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Modify the arguments
    const argsEditorEdit = userSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditorEdit.waitFor({ state: "visible" });
    await argsEditorEdit.fill('{"param": "updated_value"}');

    // Save
    await saveAndWaitForRedirect(page);

    // CRITICAL: Verify ID is still there after edit
    await expect(page.getByText(toolId)).toBeVisible();
    await expect(page.getByText("updated_value")).toBeVisible();

    // Delete tool call
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    await saveAndWaitForRedirect(page);

    // Verify tool call removed
    await expect(page.getByText(toolId)).not.toBeVisible();
  });
});

test.describe("User Message - Tool Result Blocks", () => {
  test("should add, edit, and delete tool result block in user message", async ({
    page,
  }) => {
    // Create datapoint from inference without user_schema
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Add tool result to user message
    const userSection = page.getByTestId("message-user").first();

    const addToolResultButton = userSection
      .getByRole("button", {
        name: "Tool Result",
      })
      .last();
    await expect(addToolResultButton).toBeVisible();
    await addToolResultButton.click();

    // Fill tool result details
    const resultId = "result_" + v7();
    const resultName =
      "test_result_" + Math.random().toString(36).substring(2, 10);
    const resultValue = "Result content " + v7();

    const nameInput = userSection.locator("input").first();
    await nameInput.fill(resultName);

    const idInput = userSection.locator("input").nth(1);
    await idInput.fill(resultId);

    const resultEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await resultEditor.fill(resultValue);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify tool result was added
    await expect(page.getByText(resultName)).toBeVisible();
    await expect(page.getByText(resultId)).toBeVisible();
    await expect(page.getByText(resultValue)).toBeVisible();

    // Step 2: Edit the tool result content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const resultEditor2 = userSection
      .locator("div[contenteditable='true']")
      .last();
    await resultEditor2.waitFor({ state: "visible" });

    const resultValue2 = "Updated result " + v7();
    await resultEditor2.fill(resultValue2);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify edited content
    await expect(page.getByText(resultValue2)).toBeVisible();
    await expect(page.getByText(resultValue)).not.toBeVisible();

    // Step 3: Delete tool result
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    await saveAndWaitForRedirect(page);

    // Verify tool result was deleted
    await expect(page.getByText(resultValue2)).not.toBeVisible();
    await expect(page.getByText(resultValue)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(resultValue2)).not.toBeVisible();
  });
});

test.describe("User Message - Template Blocks", () => {
  test("should delete, re-add, and edit template block in user message", async ({
    page,
  }) => {
    // Create datapoint from custom_template_test function which has named templates
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    // Step 1: Delete the text content, add a template
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons
    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add a template (greeting_template is defined in custom_template_test)
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await expect(addTemplateButton).toBeVisible();
    await addTemplateButton.click();

    // Select template name
    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("greeting_template");

    // Fill template arguments (must match greeting_template schema: name, place, day_of_week)
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    const templateValue1 = v7();
    const templateJson1 = JSON.stringify(
      { name: "Alice", place: templateValue1, day_of_week: "Monday" },
      null,
      2,
    );
    await templateEditor.fill(templateJson1);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 2: Verify template was added
    await expect(page.getByText(templateValue1)).toBeVisible();

    // Step 3: Edit the template arguments
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const templateEditor2 = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor2.waitFor({ state: "visible" });

    const templateValue2 = v7();
    const templateJson2 = JSON.stringify(
      { name: "Bob", place: templateValue2, day_of_week: "Tuesday" },
      null,
      2,
    );
    await templateEditor2.fill(templateJson2);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify template was edited
    await expect(page.getByText(templateValue2)).toBeVisible();
    await expect(page.getByText(templateValue1)).not.toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await expect(page.getByText(templateValue2)).toBeVisible();
  });
});

test.describe("User Message - Thought Blocks", () => {
  test("should add, edit, and delete thought block in user message", async ({
    page,
  }) => {
    // Create datapoint from inference without user_schema (answer_question function)
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate by waiting for Save button
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Find a user message section and add thought block
    const userSection = page.getByTestId("message-user").first();

    const addThoughtButton = userSection
      .getByRole("button", { name: "Thought" })
      .last();
    await expect(addThoughtButton).toBeVisible();
    await addThoughtButton.click();

    // Find the newly added thought editor (last contenteditable in user section)
    const thoughtEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor.waitFor({ state: "visible" });

    // Fill with content
    const thoughtContent = v7();
    await thoughtEditor.fill(`User thought content: ${thoughtContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify thought block was added
    await expect(page.getByText(thoughtContent)).toBeVisible();

    // Step 2: Edit the thought block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    // Find the thought editor again
    const thoughtEditor2 = userSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor2.waitFor({ state: "visible" });

    const thoughtContent2 = v7();
    await thoughtEditor2.fill(`Edited user thought: ${thoughtContent2}`);

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

    // Find delete button for the newly added content block (last one)
    const deleteButton = userSection
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
// Assistant Message Content Block Tests
// ============================================================================

test.describe("Assistant Message - Text Blocks", () => {
  test("should add, edit, and delete text block in assistant message", async ({
    page,
  }) => {
    // Create datapoint from inference without assistant message in input
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // First, add an assistant message to the input
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Find the newly added assistant message section
    const assistantSection = page.getByTestId("message-assistant").first();

    // Wait for the assistant section to be fully rendered
    await assistantSection
      .getByRole("button", { name: "Text" })
      .first()
      .waitFor({ state: "visible" });

    // Expand all "Show more"
    await expandShowMoreIfPresent(page);

    // Now add text to the assistant message
    const addTextButton = assistantSection
      .getByRole("button", {
        name: "Text",
      })
      .last();
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    const textEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor.waitFor({ state: "visible" });

    const textContent = v7();
    await textEditor.fill(`Assistant text: ${textContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify text block was added
    await expect(page.getByText(textContent)).toBeVisible();

    // Step 2: Edit the text block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const textEditor2 = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await textEditor2.waitFor({ state: "visible" });

    const textContent2 = v7();
    await textEditor2.fill(`Edited assistant text: ${textContent2}`);

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

    const deleteButton = assistantSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

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

test.describe("Assistant Message - Tool Call Blocks", () => {
  test("should add, edit, and delete tool call block in assistant message", async ({
    page,
  }) => {
    // Create datapoint from inference without assistant message in input
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // First, add an assistant message to the input
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Find the newly added assistant message section
    const assistantSection = page.getByTestId("message-assistant").first();

    // Wait for the assistant section to be fully rendered
    await assistantSection
      .getByRole("button", { name: "Tool Call" })
      .first()
      .waitFor({ state: "visible" });

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Add tool call to assistant message
    const addToolCallButton = assistantSection
      .getByRole("button", {
        name: "Tool Call",
      })
      .last();
    await expect(addToolCallButton).toBeVisible();
    await addToolCallButton.click();

    // Fill in tool call details
    const toolId = "tool_" + v7();
    const toolName = "test_tool_" + Math.random().toString(36).substring(2, 10);

    // Find the tool call inputs (should have Name, ID, Arguments fields)
    const nameInput = assistantSection
      .getByPlaceholder("tool_name")
      .or(assistantSection.locator('input[value=""]'))
      .first();
    await nameInput.fill(toolName);

    const idInput = assistantSection.locator("input").nth(1);
    await idInput.fill(toolId);

    const argsEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditor.fill('{"param": "value"}');

    // Save
    await saveAndWaitForRedirect(page);

    // Verify tool call visible (check for tool name and ID)
    await expect(page.getByText(toolName)).toBeVisible();
    await expect(page.getByText(toolId)).toBeVisible();

    // Reload and verify persistence
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await expect(page.getByText(toolName)).toBeVisible();
    await expect(page.getByText(toolId)).toBeVisible();

    // Edit again and verify ID is preserved
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    // Modify the arguments
    const argsEditorEdit = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await argsEditorEdit.waitFor({ state: "visible" });
    await argsEditorEdit.fill('{"param": "updated_value"}');

    // Save
    await saveAndWaitForRedirect(page);

    // CRITICAL: Verify ID is still there after edit
    await expect(page.getByText(toolId)).toBeVisible();
    await expect(page.getByText("updated_value")).toBeVisible();

    // Delete tool call
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = assistantSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

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

test.describe("Assistant Message - Tool Result Blocks", () => {
  test("should add, edit, and delete tool result block in assistant message", async ({
    page,
  }) => {
    // Create datapoint from inference without assistant message in input
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // First, add an assistant message to the input
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Find the newly added assistant message section
    const assistantSection = page.getByTestId("message-assistant").first();

    // Wait for the assistant section to be fully rendered
    await assistantSection
      .getByRole("button", { name: "Tool Result" })
      .first()
      .waitFor({ state: "visible" });

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Add tool result to assistant message
    const addToolResultButton = assistantSection
      .getByRole("button", {
        name: "Tool Result",
      })
      .last();
    await expect(addToolResultButton).toBeVisible();
    await addToolResultButton.click();

    // Fill tool result details
    const resultId = "result_" + v7();
    const resultName =
      "test_result_" + Math.random().toString(36).substring(2, 10);
    const resultValue = "Result content " + v7();

    const nameInput = assistantSection.locator("input").first();
    await nameInput.fill(resultName);

    const idInput = assistantSection.locator("input").nth(1);
    await idInput.fill(resultId);

    const resultEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await resultEditor.fill(resultValue);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify tool result was added
    await expect(page.getByText(resultName)).toBeVisible();
    await expect(page.getByText(resultId)).toBeVisible();
    await expect(page.getByText(resultValue)).toBeVisible();

    // Step 2: Edit the tool result content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const resultEditor2 = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await resultEditor2.waitFor({ state: "visible" });

    const resultValue2 = "Updated result " + v7();
    await resultEditor2.fill(resultValue2);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify edited content
    await expect(page.getByText(resultValue2)).toBeVisible();
    await expect(page.getByText(resultValue)).not.toBeVisible();

    // Step 3: Delete tool result
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = assistantSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    await saveAndWaitForRedirect(page);

    // Verify tool result was deleted
    await expect(page.getByText(resultValue2)).not.toBeVisible();
    await expect(page.getByText(resultValue)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(resultValue2)).not.toBeVisible();
  });
});

test.describe("Assistant Message - Template Blocks", () => {
  test("should add, edit, and delete template block in assistant message", async ({
    page,
  }) => {
    // Create datapoint from custom_template_test function which has named templates
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // First, add an assistant message to the input
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Find the newly added assistant message section
    const assistantSection = page.getByTestId("message-assistant").first();

    // Wait for the assistant section to be fully rendered
    await assistantSection
      .getByRole("button", { name: "Template" })
      .first()
      .waitFor({ state: "visible" });

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Add a template (greeting_template is defined in custom_template_test)
    const addTemplateButton = assistantSection
      .getByRole("button", { name: "Template" })
      .last();
    await expect(addTemplateButton).toBeVisible();
    await addTemplateButton.click();

    // Select template name
    const templateNameInput = assistantSection
      .locator('input[type="text"]')
      .first();
    await templateNameInput.fill("greeting_template");

    // Fill template arguments (must match greeting_template schema: name, place, day_of_week)
    const templateEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    const templateValue1 = v7();
    const templateJson1 = JSON.stringify(
      { name: "Alice", place: templateValue1, day_of_week: "Monday" },
      null,
      2,
    );
    await templateEditor.fill(templateJson1);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify template was added
    await expect(page.getByText(templateValue1)).toBeVisible();

    // Step 2: Edit the template arguments
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const templateEditor2 = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor2.waitFor({ state: "visible" });

    const templateValue2 = v7();
    const templateJson2 = JSON.stringify(
      { name: "Bob", place: templateValue2, day_of_week: "Tuesday" },
      null,
      2,
    );
    await templateEditor2.fill(templateJson2);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify template was edited
    await expect(page.getByText(templateValue2)).toBeVisible();
    await expect(page.getByText(templateValue1)).not.toBeVisible();

    // Step 3: Delete the template
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const deleteButton = assistantSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

    // Save
    await saveAndWaitForRedirect(page);

    // Verify template was deleted
    await expect(page.getByText(templateValue2)).not.toBeVisible();

    // Reload and verify deletion persists
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    // Wait for Edit button to ensure page is fully loaded before checking negative assertion
    await page
      .getByRole("button", { name: "Edit" })
      .waitFor({ state: "visible" });
    await expect(page.getByText(templateValue2)).not.toBeVisible();
  });
});

test.describe("Assistant Message - Thought Blocks", () => {
  test("should add, edit, and delete thought block in assistant message", async ({
    page,
  }) => {
    // Create datapoint from inference without assistant message
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // First, add an assistant message to the input
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Find the newly added assistant message section
    const assistantSection = page.getByTestId("message-assistant").first();

    // Wait for the assistant section to be fully rendered
    await assistantSection
      .getByRole("button", { name: "Thought" })
      .first()
      .waitFor({ state: "visible" });

    // Expand all "Show more" buttons if they exist (content might be collapsed)
    await expandShowMoreIfPresent(page);

    // Now add thought to the assistant message
    const addThoughtButton = assistantSection
      .getByRole("button", {
        name: "Thought",
      })
      .last();
    await expect(addThoughtButton).toBeVisible();
    await addThoughtButton.click();

    const thoughtEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor.waitFor({ state: "visible" });

    const thoughtContent = v7();
    await thoughtEditor.fill(`Assistant thought: ${thoughtContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Step 1: Verify thought block was added
    await expect(page.getByText(thoughtContent)).toBeVisible();

    // Step 2: Edit the thought block content
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons again
    await expandShowMoreIfPresent(page);

    const thoughtEditor2 = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    await thoughtEditor2.waitFor({ state: "visible" });

    const thoughtContent2 = v7();
    await thoughtEditor2.fill(`Edited assistant thought: ${thoughtContent2}`);

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

    const deleteButton = assistantSection
      .getByRole("button", { name: "Delete content block" })
      .last();
    await deleteButton.click();

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

// ============================================================================
// Message-Level Operations
// ============================================================================

test.describe("Message Operations", () => {
  test("should add and delete user message", async ({ page }) => {
    // Create datapoint
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons to avoid gradient overlay blocking clicks
    await expandShowMoreIfPresent(page);

    // Count existing user messages
    const userMessagesInitial = await page.getByTestId("message-user").count();

    // Add new user message
    const addUserButton = page.getByRole("button", { name: "User Message" });
    await expect(addUserButton).toBeVisible();
    await addUserButton.click();

    // Verify new message appeared
    await expect(page.getByTestId("message-user")).toHaveCount(
      userMessagesInitial + 1,
    );

    // Expand "Show more" again as content might have grown
    await expandShowMoreIfPresent(page);

    // Add content to new message
    const newUserSection = page.getByTestId("message-user").last();

    const addTextButton = newUserSection
      .getByRole("button", { name: "Text" })
      .last();
    await addTextButton.click();

    const textEditor = newUserSection
      .locator("div[contenteditable='true']")
      .last();
    const newMessageContent = v7();
    await textEditor.fill(`New user message: ${newMessageContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify
    await expect(page.getByText(newMessageContent)).toBeVisible();

    // Delete the message
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Find and click delete message button (not content block delete)
    const deleteMessageButton = page
      .getByRole("button", { name: "Delete message" })
      .last();
    await deleteMessageButton.click();

    await saveAndWaitForRedirect(page);

    // Verify message removed
    await expect(page.getByText(newMessageContent)).not.toBeVisible();
  });

  test("should add and delete assistant message", async ({ page }) => {
    // Create datapoint
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons to avoid gradient overlay blocking clicks
    await expandShowMoreIfPresent(page);

    // Count existing assistant messages
    const assistantMessagesInitial = await page
      .getByTestId("message-assistant")
      .count();

    // Add assistant message
    const addAssistantButton = page.getByRole("button", {
      name: "Assistant Message",
    });
    await expect(addAssistantButton).toBeVisible();
    await addAssistantButton.click();

    // Verify new message appeared
    await expect(page.getByTestId("message-assistant")).toHaveCount(
      assistantMessagesInitial + 1,
    );

    // Expand "Show more" again as content might have grown
    await expandShowMoreIfPresent(page);

    // Add content to assistant message
    const assistantSection = page.getByTestId("message-assistant").last();

    const addTextButton = assistantSection
      .getByRole("button", {
        name: "Text",
      })
      .last();
    await addTextButton.click();

    const textEditor = assistantSection
      .locator("div[contenteditable='true']")
      .last();
    const assistantContent = v7();
    await textEditor.fill(`New assistant message: ${assistantContent}`);

    // Save
    await saveAndWaitForRedirect(page);

    // Verify
    await expect(page.getByText(assistantContent)).toBeVisible();

    // Delete
    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    const deleteMessageButton = page
      .getByRole("button", { name: "Delete message" })
      .last();
    await deleteMessageButton.click();

    await saveAndWaitForRedirect(page);

    // Verify removed
    await expect(page.getByText(assistantContent)).not.toBeVisible();
  });
});

// ============================================================================
// Action Button Tests
// ============================================================================

test.describe("Delete Action Buttons", () => {
  test("should show and hide delete buttons based on edit mode", async ({
    page,
  }) => {
    // Create datapoint
    await createDatapointFromInference(page, {
      inferenceId: "0196374b-0d7a-7a22-b2d2-598a14f2eacc",
    });

    // In read-only mode, delete buttons should not be visible
    await expect(
      page.getByRole("button", { name: "Delete content block" }),
    ).not.toBeVisible();
    await expect(
      page.getByRole("button", { name: "Delete message" }),
    ).not.toBeVisible();

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons to avoid gradient overlay blocking clicks
    await expandShowMoreIfPresent(page);

    // In edit mode, delete buttons should be visible
    // (There should be at least one delete message button for existing messages)
    const deleteMessageButtons = page.getByRole("button", {
      name: "Delete message",
    });
    await expect(deleteMessageButtons.first()).toBeVisible();

    // Add a content block and verify delete button appears
    const userSection = page.getByTestId("message-user").first();

    const addTextButton = userSection.getByRole("button", { name: "Text" });
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();

    // Delete content block button should now be visible
    const deleteContentButtons = page.getByRole("button", {
      name: "Delete content block",
    });
    await expect(deleteContentButtons.first()).toBeVisible();

    // Cancel/Save and verify buttons hidden again
    await page.getByRole("button", { name: "Cancel" }).click();

    await expect(
      page.getByRole("button", { name: "Delete content block" }),
    ).not.toBeVisible();
  });
});

// ============================================================================
// Edge Cases
// ============================================================================

test.describe("Edge Cases", () => {
  test("should handle mixed content blocks in one message", async ({
    page,
  }) => {
    // Create datapoint
    await createDatapointFromInference(page, {
      inferenceId: "0196a0e8-a760-7a90-9f8a-8925365133b6",
    });

    // Enter edit mode
    await page.getByRole("button", { name: "Edit" }).click();

    // Wait for edit mode to activate
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Expand all "Show more" buttons to avoid gradient overlay blocking clicks
    await expandShowMoreIfPresent(page);

    // Add multiple content blocks to one message
    const userSection = page.getByTestId("message-user").first();

    // Add text
    const addTextButton = userSection
      .getByRole("button", { name: "Text" })
      .last();
    await expect(addTextButton).toBeVisible();
    await addTextButton.click();
    const textEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    const textContent = v7();
    await textEditor.fill(`Text: ${textContent}`);

    // Add tool call
    const addToolCallButton = userSection.getByRole("button", {
      name: "Tool Call",
    });
    await expect(addToolCallButton).toBeVisible();
    await addToolCallButton.click();

    const toolName =
      "mixed_tool_" + Math.random().toString(36).substring(2, 10);
    const toolInputs = userSection.locator('input[type="text"]');
    await toolInputs.nth(0).fill(toolName);
    await toolInputs.nth(1).fill("tool_id_123");

    const toolArgsEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await toolArgsEditor.fill('{"key": "value"}');

    // Save
    await saveAndWaitForRedirect(page);

    // Verify both content blocks are visible
    await expect(page.getByText(textContent)).toBeVisible();
    await expect(page.getByText(toolName)).toBeVisible();

    // Reload and verify
    await page.reload();
    await page.waitForLoadState("networkidle", { timeout: 5000 });
    await expect(page.getByText(textContent)).toBeVisible();
    await expect(page.getByText(toolName)).toBeVisible();
  });
});

// ============================================================================
// Error Cases - Template Validation
// ============================================================================

test.describe("Error Cases - Template Validation", () => {
  test("should reject template with non-existent template name", async ({
    page,
  }) => {
    // Create datapoint from custom_template_test function which has named templates:
    // greeting_template, analysis_prompt, fun_fact_topic
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add a template with a non-existent name
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await expect(addTemplateButton).toBeVisible();
    await addTemplateButton.click();

    // Use a template name that doesn't exist
    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("nonexistent_template");

    // Provide valid JSON (but for a template that doesn't exist)
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(
      JSON.stringify({ name: "Alice", place: "NYC" }, null, 2),
    );

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    // Wait briefly to see if an error appears or save fails
    await page.waitForTimeout(1000);

    // Verify we're still on the same page (didn't redirect) OR an error message appeared
    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });

  test("should reject template missing required fields", async ({ page }) => {
    // greeting_template schema requires: name, place, day_of_week (all required)
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add template
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await addTemplateButton.click();

    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("greeting_template");

    // Missing required fields: place and day_of_week
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(JSON.stringify({ name: "Alice" }, null, 2));

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|required/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });

  test("should reject template with wrong field types", async ({ page }) => {
    // greeting_template schema expects strings, but we'll provide a number
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add template
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await addTemplateButton.click();

    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("greeting_template");

    // Wrong type: name should be string, but we provide number
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(
      JSON.stringify(
        { name: 123, place: "NYC", day_of_week: "Monday" },
        null,
        2,
      ),
    );

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|type/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });

  test("should reject template with extra fields", async ({ page }) => {
    // greeting_template schema has additionalProperties: false
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add template
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await addTemplateButton.click();

    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("greeting_template");

    // Extra field that's not in the schema
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(
      JSON.stringify(
        {
          name: "Alice",
          place: "NYC",
          day_of_week: "Monday",
          extra_field: "not_allowed",
        },
        null,
        2,
      ),
    );

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|additional/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });

  test("should reject template with malformed JSON", async ({ page }) => {
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add template
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await addTemplateButton.click();

    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("greeting_template");

    // Malformed JSON - missing closing brace
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(
      '{"name": "Alice", "place": "NYC", "day_of_week": "Monday"',
    );

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|json/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });

  test("should reject template with empty name", async ({ page }) => {
    await createDatapointFromInference(page, {
      inferenceId: "019a0881-7437-7495-b506-782079c593bf",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    await expandShowMoreIfPresent(page);

    const userSection = page.getByTestId("message-user").first();

    // Delete the existing text block
    const deleteTextButton = userSection
      .getByRole("button", { name: "Delete content block" })
      .first();
    await deleteTextButton.click();

    // Add template
    const addTemplateButton = userSection
      .getByRole("button", { name: "Template" })
      .last();
    await addTemplateButton.click();

    // Leave template name empty
    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("");

    // Provide valid JSON
    const templateEditor = userSection
      .locator("div[contenteditable='true']")
      .last();
    await templateEditor.fill(
      JSON.stringify(
        { name: "Alice", place: "NYC", day_of_week: "Monday" },
        null,
        2,
      ),
    );

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|empty|required/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });
});

// ============================================================================
// Error Cases - System Template Validation
// ============================================================================

test.describe("Error Cases - System Template Validation", () => {
  test("should reject system template with malformed JSON", async ({
    page,
  }) => {
    // Use answer_question function which has system_schema
    await createDatapointFromInference(page, {
      inferenceId: "01968d06-392d-7451-b32c-e77ed6b13146",
    });

    await page.getByRole("button", { name: "Edit" }).click();
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();

    // Delete existing system message
    const deleteSystemButton = page.getByRole("button", {
      name: "Delete system",
    });
    await expect(deleteSystemButton).toBeVisible();
    await deleteSystemButton.click();

    // Add system template
    const systemSection = page.getByTestId("message-system");
    const addTemplateButton = systemSection.getByRole("button", {
      name: "Template",
    });
    await expect(addTemplateButton).toBeVisible();
    await addTemplateButton.click();

    // Provide malformed JSON - missing closing brace
    const templateEditor = page.locator("div[contenteditable='true']").first();
    await templateEditor.waitFor({ state: "visible" });
    await templateEditor.fill('{"secret": "value"');

    // Try to save - should fail
    const currentUrl = page.url();
    await page.getByRole("button", { name: "Save" }).click();

    await page.waitForTimeout(1000);

    const stillOnSamePage = page.url() === currentUrl;
    const errorVisible = await page
      .getByText(/error|invalid|failed|json/i)
      .isVisible()
      .catch(() => false);

    expect(stillOnSamePage || errorVisible).toBeTruthy();
  });
});
