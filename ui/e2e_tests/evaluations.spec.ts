import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto("/evaluations");
  await expect(page.getByText("New Run")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("push the new run button, launch an eval", async ({ page }) => {
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByText("Select an evaluation").click();
  await page.waitForTimeout(500);
  await page
    .locator('select[name="eval_name"]')
    .selectOption("entity_extraction");
  await page.mouse.click(10, 10);
  await page.getByText("Select a variant").click();
  await page.waitForTimeout(500);
  await page
    .locator('select[name="variant_name"]')
    .selectOption("gpt4o_mini_initial_prompt");
  await page.mouse.click(10, 10);
  await page.getByRole("button", { name: "Launch" }).click();
  await page.waitForTimeout(5000);

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
