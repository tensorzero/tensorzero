import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction/01939a16-b258-71e1-a467-183001c1952c?evaluation_run_ids=0195c501-8e6b-76f2-aa2c-d7d379fe22a5,0195aef8-36bf-7c02-b8a2-40d78049a4a0",
  );
  await expect(page.getByText("Datapoint")).toBeVisible();
  await expect(page.getByText("llama_8b_initial_prompt")).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
