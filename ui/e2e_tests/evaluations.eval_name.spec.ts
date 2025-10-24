import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063%2C0196367b-c0bb-7f90-b651-f90eb9fba8f3",
  );
  await expect(page.getByText("Input")).toBeVisible();
  await expect(page.getByText("llama_8b_initial_prompt")).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("Navigate through ragged evaluation results", async ({ page }) => {
  await page.goto(
    "/evaluations/haiku?evaluation_run_ids=0196374b-04a3-7013-9049-e59ed5fe3f74%2C01963691-9d3c-7793-a8be-3937ebb849c1",
  );

  // Wait for the table row containing the giant topic to be visible
  await expect(page.locator('td:has-text("sheet")').first()).toBeVisible();

  // Sleep for a bit to ensure the table is fully loaded
  await page.waitForTimeout(500);
  // Click on the first row that contains "sheet"
  await page.locator('td:has-text("sheet")').first().click();

  // Verify the URL contains the correct datapoint ID and evaluation run IDs
  await expect(page).toHaveURL(
    "/evaluations/haiku/0196374a-d03f-7420-9da5-1561cba71ddb?evaluation_run_ids=0196374b-04a3-7013-9049-e59ed5fe3f74",
  );
  await expect(page.getByText("soft, light, covering")).toHaveCount(2);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to add float feedback from the evaluation result page", async ({
  page,
}) => {
  await page.goto(
    "/evaluations/entity_extraction?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Find the first table row and hover over the rightmost cell (last td)
  const firstRow = page.locator("tbody tr").first();
  const rightmostCell = firstRow.locator("td").last();
  await rightmostCell.hover();

  // Click on the "pencil" icon
  await page.locator("svg.lucide-pencil").first().click();

  // Wait for the modal to appear
  await page.locator('div[role="dialog"]').waitFor({
    state: "visible",
  });

  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;

  // Fill in the float value
  await page
    .getByRole("spinbutton", { name: "Value" })
    .fill(randomFloat.toString());

  // Click on the "Save" button
  await page.locator('button[type="submit"]').click();

  // Assert that the float value is displayed
  await expect(page.getByText(randomFloat.toString())).toBeVisible();

  // Check that the new URL has search params `newFeedbackId` and `newJudgeDemonstrationId`
  const url = new URL(page.url());
  expect(url.searchParams.get("newFeedbackId")).toBeDefined();
  expect(url.searchParams.get("newJudgeDemonstrationId")).toBeDefined();
});

test("should be able to add boolean feedback from the evaluation result page", async ({
  page,
}) => {
  await page.goto(
    "/evaluations/haiku?evaluation_run_ids=0196367a-702c-75f3-b676-d6ffcc7370a1",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Find the first table row and hover over the rightmost cell (last td)
  const firstRow = page.locator("tbody tr").first();
  const rightmostCell = firstRow.locator("td").last();
  await rightmostCell.hover();

  // Click on the "pencil" icon
  await page.locator("svg.lucide-pencil").first().click();

  // Wait for the modal to appear
  await page.locator('div[role="dialog"]').waitFor({
    state: "visible",
  });

  // Click on the "True" button
  await page.getByRole("radio", { name: "True" }).click();

  // Wait for a little bit
  await page.waitForTimeout(500);

  // Click on the "Save" button
  await page.locator('button[type="submit"]').click();

  // Wait for the modal to disappear
  await page.locator('div[role="dialog"]').waitFor({
    state: "hidden",
  });

  // We don't assert that the bool value is displayed since there are gonna be many trues and falses.

  // Check that the new URL has search params `newFeedbackId` and `newJudgeDemonstrationId`
  const url = new URL(page.url());
  expect(url.searchParams.get("newFeedbackId")).toBeDefined();
  expect(url.searchParams.get("newJudgeDemonstrationId")).toBeDefined();
});
