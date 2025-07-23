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
