import { test, expect } from "@playwright/test";

test("should show the episode detail page", async ({ page }) => {
  await page.goto(
    "/observability/episodes/0195c501-d7c4-7b50-a856-02a9f00144bf",
  );
  // The function name should be visible
  await expect(
    page.getByText("tensorzero::llm_judge::entity_extraction::count_sports"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
