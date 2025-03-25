import { test, expect } from "@playwright/test";

test("should show the inference detail page", async ({ page }) => {
  await page.goto(
    "/observability/inferences/0195aef8-3eaa-7dc2-9376-8dde217649e8",
  );
  // The episode ID should be visible
  await expect(
    page.getByText("0195aef8-3eaa-7dc2-9376-8de1d8c6536b"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
