import { test, expect } from "@playwright/test";

test("should show the inference detail page", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195c501-d7c4-7b50-a856-029315411fb8",
  );
  // The episode ID should be visible
  await expect(
    page.getByText("0195c501-d7c4-7b50-a856-02a9f00144bf"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
