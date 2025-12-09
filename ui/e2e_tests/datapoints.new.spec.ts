import { test, expect } from "@playwright/test";
import { v7 as uuidv7 } from "uuid";

test.describe("New Datapoint Page", () => {
  test("should create a chat datapoint with input and output", async ({
    page,
  }) => {
    const datasetName = `test_dataset_${uuidv7()}`;
    const datapointName = `test_datapoint_${uuidv7()}`;
    const systemMessage = `System message ${uuidv7()}`;
    const topicValue = `topic_${uuidv7()}`;
    const outputText = `Output text ${uuidv7()}`;
    const tagKey = "environment";
    const tagValue = `test_value_${uuidv7()}`;

    // Navigate to new datapoint page
    await page.goto("/datapoints/new");
    await page.waitForLoadState("networkidle");

    // Verify page loaded
    await expect(
      page.getByRole("heading", { name: "New Datapoint" }),
    ).toBeVisible();

    // Select dataset (create new)
    await page.getByTestId("dataset-selector").getByRole("combobox").click();
    await page.getByPlaceholder(/dataset/i).fill(datasetName);
    await page.locator("[cmdk-item]").filter({ hasText: datasetName }).click();

    // Select function (chat type)
    await page.getByTestId("function-selector").getByRole("combobox").click();
    await page.getByPlaceholder("Find a function...").fill("write_haiku");
    await page
      .locator("[cmdk-item]")
      .filter({ hasText: "write_haiku" })
      .click();

    // Wait for form sections to appear
    await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Tags" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Metadata" })).toBeVisible();

    // Add system message using testid to scope to system section
    const systemSection = page.getByTestId("message-system");
    await systemSection.getByRole("button", { name: "Text" }).click();
    const systemEditor = systemSection
      .locator("div[contenteditable='true']")
      .first();
    await systemEditor.waitFor({ state: "visible" });
    await systemEditor.fill(systemMessage);

    // Add user message with template (write_haiku requires user_schema with "topic" field)
    await page.getByRole("button", { name: "User Message" }).click();
    const userSection = page.getByTestId("message-user").first();
    await userSection.getByRole("button", { name: "Template" }).click();
    // Fill template name
    const templateNameInput = userSection.locator('input[type="text"]').first();
    await templateNameInput.fill("user");
    // Fill template arguments
    const userTemplateEditor = userSection
      .locator("div[contenteditable='true']")
      .first();
    await userTemplateEditor.waitFor({ state: "visible" });
    await userTemplateEditor.fill(
      JSON.stringify({ topic: topicValue }, null, 2),
    );

    // Add output text - assistant section is already visible with empty output
    const chatOutput = page.getByTestId("chat-output");
    const outputAssistantSection = chatOutput.getByTestId("message-assistant");
    await outputAssistantSection.waitFor({ state: "visible" });
    await outputAssistantSection.getByRole("button", { name: "Text" }).click();
    const outputEditor = outputAssistantSection
      .locator("div[contenteditable='true']")
      .first();
    await outputEditor.waitFor({ state: "visible" });
    await outputEditor.fill(outputText);

    // Add tag
    const tagsSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Tags" }) });
    await tagsSection.getByPlaceholder("Key").fill(tagKey);
    await tagsSection.getByPlaceholder("Value").fill(tagValue);
    await tagsSection.getByRole("button", { name: "Add" }).click();

    // Fill in name
    await page.getByLabel("Name").fill(datapointName);

    // Create datapoint
    await page.getByRole("button", { name: "Create Datapoint" }).click();

    // Wait for redirect to datapoint detail page
    await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
      timeout: 10000,
    });
    await page.waitForLoadState("networkidle");

    // Verify we're on the detail page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}/datapoint/[^/]+$`),
    );

    // Verify content persists
    await expect(page.getByText(systemMessage)).toBeVisible();
    await expect(page.getByText(topicValue)).toBeVisible();
    await expect(page.getByText(outputText)).toBeVisible();
    await expect(page.getByText(tagKey)).toBeVisible();
    await expect(page.getByText(tagValue)).toBeVisible();
    await expect(page.getByText(datapointName)).toBeVisible();

    // Verify no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should create a JSON datapoint with custom output and schema", async ({
    page,
  }) => {
    const datasetName = `test_dataset_${uuidv7()}`;
    const systemMessage = `System message ${uuidv7()}`;
    const userMessage = `User message ${uuidv7()}`;
    const uniqueValue = uuidv7();
    const outputJson = JSON.stringify(
      {
        person: ["John Doe"],
        organization: ["Acme Corp"],
        location: ["New York"],
        miscellaneous: [uniqueValue],
      },
      null,
      2,
    );

    // Navigate to new datapoint page
    await page.goto("/datapoints/new");
    await page.waitForLoadState("networkidle");

    // Select dataset (create new)
    await page.getByTestId("dataset-selector").getByRole("combobox").click();
    await page.getByPlaceholder(/dataset/i).fill(datasetName);
    await page.locator("[cmdk-item]").filter({ hasText: datasetName }).click();

    // Select function (JSON type)
    await page.getByTestId("function-selector").getByRole("combobox").click();
    await page.getByPlaceholder("Find a function...").fill("extract_entities");
    await page
      .locator("[cmdk-item]")
      .filter({ hasText: "extract_entities" })
      .click();

    // Wait for form sections to appear
    await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();

    // Add system message
    const systemSection = page.getByTestId("message-system");
    await systemSection.getByRole("button", { name: "Text" }).click();
    const systemEditor = systemSection
      .locator("div[contenteditable='true']")
      .first();
    await systemEditor.waitFor({ state: "visible" });
    await systemEditor.fill(systemMessage);

    // Add user message with text (extract_entities doesn't require user schema)
    await page.getByRole("button", { name: "User Message" }).click();
    const userSection = page.getByTestId("message-user").first();
    await userSection.getByRole("button", { name: "Text" }).click();
    const userEditor = userSection
      .locator("div[contenteditable='true']")
      .first();
    await userEditor.waitFor({ state: "visible" });
    await userEditor.fill(userMessage);

    // Find Output section (same pattern as editing-output.spec.ts)
    const outputSection = page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Output" }) });

    // Verify Schema tab is present (JSON functions have output schema)
    await expect(
      outputSection.getByRole("tab", { name: "Schema" }),
    ).toBeVisible();

    // Edit the raw output - use .last() to get the editor in the output section
    // Note: Raw Output tab is already selected by default in edit mode
    const outputEditor = outputSection
      .locator("div[contenteditable='true']")
      .last();
    await outputEditor.waitFor({ state: "visible" });
    await outputEditor.click();
    await outputEditor.fill(outputJson);

    // Verify we can see the schema tab and it has content (use exact match to avoid "additionalProperties")
    await outputSection.getByRole("tab", { name: "Schema" }).click();
    await expect(page.getByText('"properties"', { exact: true })).toBeVisible();

    // Create datapoint
    await page.getByRole("button", { name: "Create Datapoint" }).click();

    // Wait for redirect to datapoint detail page
    await page.waitForURL(`/datasets/${datasetName}/datapoint/**`, {
      timeout: 10000,
    });
    await page.waitForLoadState("networkidle");

    // Verify we're on the detail page
    await expect(page).toHaveURL(
      new RegExp(`/datasets/${datasetName}/datapoint/[^/]+$`),
    );

    // Verify input content persists
    await expect(page.getByText(systemMessage)).toBeVisible();
    await expect(page.getByText(userMessage)).toBeVisible();

    // Click Raw Output tab and verify output content
    await page.getByRole("tab", { name: "Raw Output" }).click();
    await expect(page.getByText(uniqueValue)).toBeVisible();

    // Click Schema tab and verify schema content
    await page.getByRole("tab", { name: "Schema" }).click();
    await expect(page.getByText('"properties"', { exact: true })).toBeVisible();

    // Verify no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should handle switching from JSON to chat function without crashing", async ({
    page,
  }) => {
    await page.goto("/datapoints/new");
    await page.waitForLoadState("networkidle");

    // Select dataset first
    await page.getByTestId("dataset-selector").getByRole("combobox").click();
    await page.getByPlaceholder(/dataset/i).fill("foo");
    await page.locator("[cmdk-item]").filter({ hasText: "foo" }).click();

    // Select JSON function first
    await page.getByTestId("function-selector").getByRole("combobox").click();
    await page.getByPlaceholder("Find a function...").fill("extract_entities");
    await page
      .locator("[cmdk-item]")
      .filter({ hasText: "extract_entities" })
      .click();

    // Wait for JSON output section with Schema tab
    await expect(page.getByRole("tab", { name: "Schema" })).toBeVisible();

    // Switch to chat function
    await page.getByTestId("function-selector").getByRole("combobox").click();
    await page.getByPlaceholder("Find a function...").fill("write_haiku");
    await page
      .locator("[cmdk-item]")
      .filter({ hasText: "write_haiku" })
      .click();

    // Verify chat output renders without error (no Schema tab for chat)
    await expect(page.getByRole("heading", { name: "Output" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Schema" })).not.toBeVisible();

    // Verify the chat output component is visible
    await expect(page.getByTestId("chat-output")).toBeVisible();
  });
});
