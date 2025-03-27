import { test, expect } from "@playwright/test";

test("should show the inference list page", async ({ page }) => {
  await page.goto("/observability/inferences");
  await expect(page.getByText("Inference ID")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
