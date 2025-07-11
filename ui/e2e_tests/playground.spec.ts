import test, { expect } from "@playwright/test";

test("should run inference on a dataset", async ({ page }) => {
  await page.goto("/playground");
  await expect(page.getByRole("heading", { name: "Playground" })).toBeVisible();

  // TODO Other UI assertions?

  // Select function
  // TODO should have label for function combobox, right?
  await page.getByRole("combobox").click();
  await page.getByRole("option", { name: "ask_question" }).click();

  // Select variants
  await page.getByRole("checkbox", { name: "gpt-4.1-mini" }).click();
  await page.getByRole("checkbox", { name: "gpt-4.1-nano" }).click();

  // TODO Make assertions about each variant - does it show x template...?

  // Select dataset
  await page
    .getByRole("combobox")
    .filter({ hasText: "Select a dataset" })
    .click();
  await page.getByRole("option", { name: "bar" }).click();

  // TODO Assert "No runs" initially

  const runButton = page.getByRole("button", { name: "Run" }).first();
  await runButton.click();

  // TODO Assert output is correct---

  // await runButton.click();
  // const outputButton = runButton
  //   .locator("..")
  //   .locator("..")
  //   .getByRole("button", { name: "Output" });
  // await expect(outputButton).toBeVisible();
});

test.skip("should open and close variants for a function", async ({ page }) => {
  await page.goto("/playground");

  await page.getByRole("combobox").click();
  await page.getByRole("option", { name: "ask_question" }).click();

  await page.getByRole("checkbox", { name: "gpt-4.1-mini" }).click();
  await page.getByRole("checkbox", { name: "gpt-4.1-nano" }).click();

  // Close one of them
  // TODO Click the close button for nano

  await expect(
    page.getByRole("checkbox", { name: "gpt-4.1-nano" }),
  ).not.toBeChecked();
});

test.skip("should run inference with edited user template", async () => {});

test.skip("should run inference with edited assistant template", async () => {});

// TODO Use that load/save session state
test.skip("should save layout across browser sessions", async () => {});
