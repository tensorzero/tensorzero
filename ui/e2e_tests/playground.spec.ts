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
  await expect(page.getByText("foo")).toBeVisible();
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

  // Verify that there are 4 outputs, one for each variant and each datapoint
  await expect(page.getByRole("textbox")).toHaveCount(4);

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
  await expect(page.getByText("foo")).toBeVisible();
  await expect(page.getByRole("link", { name: "baseline" })).toBeVisible();
  await expect(
    page.getByRole("link", { name: "gpt4o_mini_initial_prompt" }),
  ).toBeVisible();

  // Verify that there are 2 inputs and 2 reference outputs
  await expect(page.getByRole("heading", { name: "Input" })).toHaveCount(2);
  await expect(
    page.getByRole("heading", { name: "Reference Output" }),
  ).toHaveCount(2);

  // For JSON functions, outputs are displayed in CodeMirror editors rather than textboxes
  // Verify that there are CodeMirror editors for each variant and datapoint
  await expect(page.locator(".cm-editor")).toHaveCount(4);

  // Verify that there are no errors
  await expect(
    page.getByRole("heading", { name: "Inference Error" }),
  ).toHaveCount(0);
});

test("playground should work for image_judger function with images in input", async ({
  page,
}) => {
  // We set 'limit=1' so that we don't make parallel inference requests
  // (two of the datapoints have the sample input, and could trample on each other's
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
  await expect(page.getByText("baz")).toBeVisible();
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
