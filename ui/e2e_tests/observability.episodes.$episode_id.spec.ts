import { test, expect } from "@playwright/test";

test("should show the episode detail page", async ({ page }) => {
  await page.goto(
    "/observability/episodes/019639b3-7444-7d20-a67a-1bf97ecf132a",
  );
  // The function name should be visible
  await expect(
    page.getByText("tensorzero::llm_judge::entity_extraction::count_sports"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
