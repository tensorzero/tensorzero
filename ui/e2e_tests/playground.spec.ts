import { test, expect } from "@playwright/test";

test("playground should work for a chat function that sets 2 variants", async ({
  page,
}) => {
  await page.goto("/playground?limit=2");
  await expect(page.getByText("Select a function")).toBeVisible();

  // Select function 'write_haiku' by typing in the combobox
  await page.getByText("Select a function").click();
  await page.getByPlaceholder("Find a function...").fill("write_haiku");
  await page.getByRole("option", { name: "write_haiku" }).click();

  // Select dataset 'foo'
  await page.getByText("Select a dataset").click();
  await page.getByPlaceholder(/dataset/i).fill("foo");
  await page.getByRole("option", { name: "foo" }).click();

  // Select variant 'initial_prompt_gpt4o_mini'
  await page
    .getByPlaceholder("Filter by variant...")
    .fill("initial_prompt_gpt4o_mini");
  await page.getByRole("option", { name: "initial_prompt_gpt4o_mini" }).click();
  await page.getByRole("option", { name: "initial_prompt_haiku_3_5" }).click();

  // Verify the selections are visible
  await expect(page.getByText("write_haiku")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "foo" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "initial_prompt_gpt4o_mini" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "initial_prompt_haiku_3_5" }),
  ).toBeVisible();

  // Verify that there are 2 inputs and 2 reference outputs
  await expect(page.getByRole("heading", { name: "Input" })).toHaveCount(2);
  await expect(
    page.getByRole("heading", { name: "Reference Output" }),
  ).toHaveCount(2);

  // Verify that there are 8 outputs, one for each variant and each datapoint
  await expect(page.getByRole("textbox")).toHaveCount(8, { timeout: 10_000 });

  // Verify that there are no errors
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);
});

test("playground should work for extract_entities JSON function with 2 variants", async ({
  page,
}) => {
  await page.goto("/playground?limit=2");
  await expect(page.getByText("Select a function")).toBeVisible();

  // Select function 'extract_entities' by typing in the combobox
  await page.getByText("Select a function").click();
  await page.getByPlaceholder("Find a function...").fill("extract_entities");
  await page.getByRole("option", { name: "extract_entities" }).click();

  // Select dataset 'foo'
  await page.getByText("Select a dataset").click();
  await page.getByPlaceholder(/dataset/i).fill("foo");
  await page.getByRole("option", { name: "foo" }).click();

  // Select variants 'baseline' and 'gpt4o_mini_initial_prompt'
  await page.getByPlaceholder("Filter by variant...").fill("baseline");
  await page.getByRole("option", { name: "baseline" }).click();

  await page
    .getByPlaceholder("Filter by variant...")
    .fill("gpt4o_mini_initial_prompt");
  await page.getByRole("option", { name: "gpt4o_mini_initial_prompt" }).click();

  // Verify the selections are visible
  await expect(page.getByText("extract_entities")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "foo" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "baseline" })).toBeVisible();
  await expect(
    page.getByRole("link", { name: "gpt4o_mini_initial_prompt" }),
  ).toBeVisible();

  // Verify that there are 2 inputs and 2 reference outputs
  await expect(page.getByRole("heading", { name: "Input" })).toHaveCount(2);
  await expect(
    page.getByRole("heading", { name: "Reference Output" }),
  ).toHaveCount(2);

  // For JSON functions, we should have CodeMirror editors for:
  // - 2 Input sections (one per datapoint)
  // - 2 Reference Output sections (one per datapoint)
  // - 4 Variant Output sections (2 variants × 2 datapoints)
  // Total: 8 CodeMirror editors
  // Wait for all editors to load before counting
  await expect(page.locator(".cm-editor")).toHaveCount(8);

  // Verify that there are no errors
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);
});

test("playground should work for image_judger function with images in input", async ({
  page,
}) => {
  // We set 'limit=1' so that we don't make parallel inference requests
  // (two of the datapoints have the same input, and could trample on each other's
  // cache entries)
  await page.goto("/playground?limit=1");
  await expect(page.getByText("Select a function")).toBeVisible();

  // Select function 'image_judger' by typing in the combobox
  await page.getByText("Select a function").click();
  await page.getByPlaceholder("Find a function...").fill("image_judger");
  await page.getByRole("option", { name: "image_judger" }).click();

  // Select dataset 'baz'
  await page.getByText("Select a dataset").click();
  await page.getByPlaceholder(/dataset/i).fill("baz");
  await page.getByRole("option", { name: "baz" }).click();

  // Select variant 'honest_answer'
  await page.getByPlaceholder("Filter by variant...").fill("honest_answer");
  await page.getByRole("option", { name: "honest_answer" }).click();

  // Verify the selections are visible
  await expect(page.getByText("image_judger")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "baz" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "honest_answer" })).toBeVisible();

  // Verify that there is 1 input and 1 reference output
  await expect(page.getByRole("heading", { name: "Input" })).toHaveCount(1);
  await expect(
    page.getByRole("heading", { name: "Reference Output" }),
  ).toHaveCount(1);

  // Verify that the image is rendered in the input element
  await expect(page.locator("img")).toHaveCount(1);

  // Wait for at least one textbox containing "crab"
  // Wait for and assert at least one exists
  // Wait for at least one textbox containing "crab" to appear
  await page.getByRole("textbox").filter({ hasText: "crab" }).first().waitFor();

  // Verify that there are no errors
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);
});

test("playground should work for data with tools", async ({ page }) => {
  await page.goto(
    '/playground?functionName=multi_hop_rag_agent&datasetName=tool_call_examples&variants=%5B%7B"type"%3A"builtin"%2C"name"%3A"baseline"%7D%5D',
  );

  // Verify the selections are visible
  await expect(page.getByText("multi_hop_rag_agent")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "tool_call_examples" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "baseline" })).toBeVisible();

  // Verify that there is 1 input and 1 reference output
  await expect(page.getByRole("heading", { name: "Input" })).toHaveCount(1);
  await expect(
    page.getByRole("heading", { name: "Reference Output" }),
  ).toHaveCount(1);

  // Verify that tool calls are displayed correctly
  // Give the inference lots of time to run - assert at least one tool call since apparently sometimes the model outputs 2
  await expect(
    page
      .getByTestId("datapoint-playground-output")
      .getByText("Tool Call")
      .first(),
  ).toBeVisible({ timeout: 30_000 });

  // Verify that at least one tool call has the expected fields
  await expect(page.getByText("Name").first()).toBeVisible();
  await expect(page.getByText("ID").first()).toBeVisible();
  await expect(page.getByText("Arguments").first()).toBeVisible();

  // Verify that there are no errors before refresh
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);

  // TODO - clicking the refresh button immediately after the inference loads doesn't seem to work
  // We should figure out what event to wait for, and remove this sleep
  await page.waitForTimeout(1000);

  // Click the refresh button to reload inference
  // Find the refresh button in the output area
  const refreshButton = page.getByTestId(
    "datapoint-playground-output-refresh-button",
  );
  await refreshButton.first().click();

  // NOTE (bad tests coverage):
  // we can't assert well that the refresh state was displayed since all the inferences are cached
  // so the response could come super fast
  // We would have to build in some delay in test mode to ensure the refresh state is displayed
  // await page.waitForTimeout(1000);
  // Since this test is flaky and blocking merges we'll remove the check for now

  // Wait for loading indicator to appear (indicates refresh started)
  // await expect(
  //   page.getByTestId("datapoint-playground-output-loading"),
  // ).toBeVisible({
  //   timeout: 5000,
  // });

  // Wait for loading indicator to disappear (indicates refresh completed)
  // await expect(
  //   page.getByTestId("datapoint-playground-output-loading"),
  // ).not.toBeVisible({
  //   timeout: 15000,
  // });

  // Verify tool calls are still displayed after refresh
  await expect(
    page
      .getByTestId("datapoint-playground-output")
      .getByText("Tool Call")
      .first(),
  ).toBeVisible({ timeout: 30_000 });

  // Verify that at least one tool call has the expected fields
  await expect(page.getByText("Name").first()).toBeVisible();
  await expect(page.getByText("ID").first()).toBeVisible();
  await expect(page.getByText("Arguments").first()).toBeVisible();

  // Verify that there are no errors after refresh
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);
});

test("editing variants works @credentials", async ({ page }) => {
  await page.goto(
    '/playground?functionName=write_haiku&datasetName=foo&variants=%5B%7B"type"%3A"builtin"%2C"name"%3A"initial_prompt_gpt4o_mini"%7D%5D',
  );

  // Verify the selections are visible
  await expect(page.getByText("write_haiku")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "foo" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "initial_prompt_gpt4o_mini" }),
  ).toBeVisible();

  // Try to edit the variant
  // First, click the edit button
  await page.getByRole("button", { name: "Edit" }).click();

  // Wait till the modal is open
  await expect(page.getByText("Variant Configuration")).toBeVisible();

  // Wait for modal animations to complete
  await page.waitForTimeout(1000);

  // edit the system prompt to say "write a haiku about the given topic. You are additional required to include the word \"obtuse\""
  // Wait for the editor content to be available and clear it
  // Target the system template editor specifically within the modal/sheet content
  const systemTemplateEditor = page
    .getByRole("dialog")
    .getByText("system")
    .locator("..")
    .locator(".cm-content")
    .first();
  await systemTemplateEditor.waitFor({ state: "visible" });

  // Select all content and replace it, using force to bypass modal overlay issues
  await systemTemplateEditor.click({ force: true });
  await page.keyboard.press("Control+a");
  await page.keyboard.type(
    'Write a haiku about the given topic. You are additionally required to include the word "obtuse".',
  );

  // save the edit
  await page.getByRole("button", { name: "Save Changes" }).click();

  // Wait for the modal to close
  await expect(page.getByText("Variant Configuration")).not.toBeVisible();

  // Wait for the inference to complete and assert that the generated output contains the word "obtuse"
  await expect(
    page.getByRole("textbox").filter({ hasText: "obtuse" }).first(),
  ).toBeVisible({ timeout: 10000 });
});

test("should navigate to inference from playground and verify tags", async ({
  page,
}) => {
  await page.goto(
    '/playground?functionName=write_haiku&datasetName=foo&variants=%5B%7B"type"%3A"builtin"%2C"name"%3A"initial_prompt_gpt4o_mini"%7D%5D&limit=1',
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Verify the selections are visible
  await expect(page.getByText("write_haiku")).toBeVisible();
  await expect(
    page.getByRole("combobox").filter({ hasText: "foo" }),
  ).toBeVisible();

  // Wait for the inference to complete - look for the inference link
  const inferenceLink = page.getByText(/View inference:/);
  await inferenceLink.waitFor({ state: "visible", timeout: 15000 });

  // Click the inference link to navigate to the inference page
  await inferenceLink.click();

  // Wait for navigation to the new inference page
  await page.waitForURL("/observability/inferences/**", {
    timeout: 5000,
  });

  // Verify we're on an inference detail page
  await expect(page.getByText("Inference", { exact: true })).toBeVisible();

  // Verify the tags section shows our UI tags
  await expect(page.getByText("tensorzero::internal")).toBeVisible();
  await expect(page.getByText("tensorzero::ui")).toBeVisible();

  // Verify the tag values are "true"
  const internalTagValue = page
    .locator("tr")
    .filter({ hasText: "tensorzero::internal" })
    .getByText("true");
  await expect(internalTagValue).toBeVisible();

  const uiTagValue = page
    .locator("tr")
    .filter({ hasText: "tensorzero::ui" })
    .getByText("true");
  await expect(uiTagValue).toBeVisible();
});
