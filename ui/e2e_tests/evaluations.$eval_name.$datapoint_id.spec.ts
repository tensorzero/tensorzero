import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction/01939a16-b258-71e1-a467-183001c1952c?evaluation_run_ids=0196368f-19bd-7082-a677-1c0bf346ff24%2C0196368e-53a8-7e82-a88d-db7086926d81",
  );
  await expect(page.getByText("Datapoint")).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toHaveCount(2);
  await expect(page.getByText("gpt4o_initial_prompt")).toHaveCount(2);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
