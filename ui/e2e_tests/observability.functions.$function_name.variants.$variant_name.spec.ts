import { test, expect } from "@playwright/test";

test("should show the variant detail page", async ({ page }) => {
  await page.goto(
    "/observability/functions/extract_entities/variants/gpt4o_mini_initial_prompt",
  );
  await expect(page.getByText("Weight")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
