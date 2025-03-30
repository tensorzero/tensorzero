import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction?eval_run_ids=0195c501-8e6b-76f2-aa2c-d7d379fe22a5%2C0195aef8-36bf-7c02-b8a2-40d78049a4a0",
  );
  await expect(page.getByText("Input")).toBeVisible();
  await expect(page.getByText("llama_8b_initial_prompt")).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("Navigate through ragged evaluation results", async ({ page }) => {
  await page.goto(
    "/evaluations/haiku/?eval_run_ids=0195c498-1cbe-7ac0-b5b2-5856741f5890%2C0195aef7-96fe-7d60-a2e6-5a6ea990c425%2C0195aef6-4ed4-7710-ae62-abb10744f153",
  );

  // Wait for the table row containing the giant topic to be visible
  await expect(page.locator('td:has-text("giant")').first()).toBeVisible();

  // Click on the first row that contains "giant"
  await page.locator('td:has-text("giant")').first().click();

  // Verify the URL contains the correct datapoint ID and eval run IDs
  await expect(page).toHaveURL(
    "evaluations/haiku/0195c497-03c2-7523-aa83-9caf73dd47d5?eval_run_ids=0195c498-1cbe-7ac0-b5b2-5856741f5890",
  );
  // Verify the page rendered properly
  await expect(page.getByText("Galaxies now dust")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
