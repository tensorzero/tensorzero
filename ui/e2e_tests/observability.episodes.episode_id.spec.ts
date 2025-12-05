import { test, expect } from "@playwright/test";

test("should show the episode detail page", async ({ page }) => {
  await page.goto(
    "/observability/episodes/0196367a-842d-74c2-9e62-67f07369b6ad",
  );
  // The function name should be visible
  await expect(
    page.getByText("tensorzero::llm_judge::haiku::topic_starts_with_f"),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to add comment feedback via the episode page", async ({
  page,
}) => {
  await page.goto(
    "/observability/episodes/01963691-b93a-7973-bcf8-9688cc02a491",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");
  // Click on the Add feedback button
  await page.getByText("Add feedback").click();

  // Open the metric combobox
  await page.getByRole("combobox", { name: "Metric" }).click();

  // Explicitly wait for the item to be visible before clicking
  const metricItemLocator = page
    .locator('div[role="dialog"]')
    .locator('div[cmdk-item=""]')
    .filter({
      hasText: "comment",
    });
  await metricItemLocator.waitFor({ state: "visible" });
  // Click on the metric in the command list
  await metricItemLocator.click();

  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;
  // Locate the dialog first
  const dialog = page.locator('div[role="dialog"]');
  // Locate the textbox within the dialog and fill it
  await dialog.getByRole("textbox").waitFor({ state: "visible" });
  // This doesn't have to be a JSON, it can be any string
  const json = `{"thinking": "hmm", "score": ${randomFloat}}`;
  await dialog.getByRole("textbox").fill(json);

  // Click the submit button
  await page.getByText("Submit Feedback").click();

  await page.waitForURL((url) => url.searchParams.has("newFeedbackId"), {
    timeout: 10000,
  });

  const newFeedbackId = new URL(page.url()).searchParams.get("newFeedbackId");
  if (!newFeedbackId) {
    throw new Error("newFeedbackId is not present in the url");
  }
  // Assert that the feedback value is visible in its table cell
  await expect(page.getByRole("cell", { name: newFeedbackId })).toBeVisible();
  // Assert that the comment is visible in the comment section
  await expect(page.getByText(json)).toBeVisible();
});
