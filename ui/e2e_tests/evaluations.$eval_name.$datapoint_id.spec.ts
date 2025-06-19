import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto(
    "/evaluations/entity_extraction/01939a16-b258-71e1-a467-183001c1952c?evaluation_run_ids=0196368f-19bd-7082-a677-1c0bf346ff24%2C0196368e-53a8-7e82-a88d-db7086926d81",
  );

  await expect(page.getByText("Datapoint", { exact: true })).toBeVisible();
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toHaveCount(2);
  await expect(page.getByText("gpt4o_initial_prompt")).toHaveCount(2);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("should be able to add float feedback from the evaluation datapoint result page", async ({
  page,
}) => {
  await page.goto(
    "/evaluations/entity_extraction/0193994e-5560-7610-a3a0-45fdd59338aa?evaluation_run_ids=0196367b-1739-7483-b3f4-f3b0a4bda063%2C0196367b-c0bb-7f90-b651-f90eb9fba8f3",
  );
  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Click on the leftmost "pencil" icon
  await page.locator("svg.lucide-pencil").first().click();

  // Wait for the modal to appear
  await page.locator('div[role="dialog"]').waitFor({
    state: "visible",
  });
  // sleep for 500ms
  await page.waitForTimeout(500);

  // Fill in the float value
  // Generate a random float between 0 and 1 with 3 decimal places
  const randomFloat = Math.floor(Math.random() * 1000) / 1000;
  await page
    .getByRole("spinbutton", { name: /Value/i })
    .fill(randomFloat.toString());
  // sleep for 500ms
  await page.waitForTimeout(500);
  // Click on the "Save" button
  await page.locator('button[type="submit"]').click();

  // Assert that the float value is displayed
  await expect(page.getByText(randomFloat.toString())).toBeVisible();

  // Check that the new URL has search params `newFeedbackId` and `newJudgeDemonstrationId`
  const url = new URL(page.url());
  expect(url.searchParams.get("newFeedbackId")).toBeDefined();
  expect(url.searchParams.get("newJudgeDemonstrationId")).toBeDefined();
});

test("should be able to add bool feedback from the evaluation datapoint result page", async ({
  page,
}) => {
  await page.goto(
    "/evaluations/haiku/01945256-c5b0-7b40-8840-caf0ee0c2c49?evaluation_run_ids=0196367a-702c-75f3-b676-d6ffcc7370a1",
  );

  // Wait for the page to load
  await page.waitForLoadState("networkidle");

  // Click on the leftmost "pencil" icon
  await page.locator("svg.lucide-pencil").first().click();

  // Wait for the modal to appear
  await page.locator('div[role="dialog"]').waitFor({
    state: "visible",
  });

  // Click on the "True" button
  await page.getByRole("radio", { name: /True/i }).click();

  // Sleep for 100ms
  await new Promise((resolve) => setTimeout(resolve, 100));

  // Click on the "Save" button
  await page.locator('button[type="submit"]').click();

  // Wait for the page to load
  await new Promise((resolve) => setTimeout(resolve, 500));

  // Assert that the bool value is displayed
  await expect(page.getByText("True")).toBeVisible();

  // Check that the new URL has search params `newFeedbackId` and `newJudgeDemonstrationId`
  const url = new URL(page.url());
  expect(url.searchParams.get("newFeedbackId")).toBeDefined();
  expect(url.searchParams.get("newJudgeDemonstrationId")).toBeDefined();
});
