import { test, expect } from "@playwright/test";

test("should show the episode list page", async ({ page }) => {
  await page.goto("/observability/episodes");
  await expect(page.getByText("Episode ID")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should not allow paging left when first loaded with no query parameters", async ({
  page,
}) => {
  await page.goto("/observability/episodes");

  // Wait for the page to load
  await expect(page.getByText("Episode ID")).toBeVisible();

  // Find the left/previous pagination button more specifically
  // Look for buttons in a flex container (the pagination container) with ChevronLeft
  const paginationContainer = page.locator(
    "div.mt-4.flex.items-center.justify-center.gap-2",
  );
  const prevButton = paginationContainer.locator("button").first();

  await expect(prevButton).toBeDisabled();
});

test("should not allow paging right when when at the end of episodes", async ({
  page,
}) => {
  // This is taken from the fixtures database.
  const minEpisodeId = "0192ced0-947e-74b3-a3d7-02fd2c54d638";
  await page.goto("/observability/episodes?before=" + minEpisodeId);

  // Wait for the page to load
  await expect(page.getByText("Episode ID")).toBeVisible();

  // Find the right/next pagination button more specifically
  // Look for buttons in a flex container (the pagination container) with ChevronRight
  const paginationContainer = page.locator(
    "div.mt-4.flex.items-center.justify-center.gap-2",
  );
  const nextButton = paginationContainer.locator("button").last();

  await expect(nextButton).toBeDisabled();
});
