import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063%2C0196367b-c0bb-7f90-b651-f90eb9fba8f3",
  );
  await expect(page.getByText("Input")).toBeVisible();
  await expect(page.getByText("llama_8b_initial_prompt")).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("Navigate through ragged evaluation results", async ({ page }) => {
  await page.goto(
    "/evaluations/haiku?evaluation_run_ids=0196374b-04a3-7013-9049-e59ed5fe3f74%2C01963691-9d3c-7793-a8be-3937ebb849c1",
  );

  // Wait for the table row containing the giant topic to be visible
  await expect(page.locator('td:has-text("sheet")').first()).toBeVisible();

  // Click on the first row that contains "giant"
  await page.locator('td:has-text("sheet")').first().click();

  // Verify the URL contains the correct datapoint ID and evaluation run IDs
  await expect(page).toHaveURL(
    "/evaluations/haiku/0196374a-d03f-7420-9da5-1561cba71ddb?evaluation_run_ids=0196374b-04a3-7013-9049-e59ed5fe3f74",
  );
  await expect(page.getByText("soft, light, covering")).toHaveCount(2);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
