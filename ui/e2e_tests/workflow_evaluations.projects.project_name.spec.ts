import { test, expect } from "@playwright/test";

test("workflow evaluation project page should render and show correct information", async ({
  page,
}) => {
  await page.goto("/workflow_evaluations/projects/beerqa-agentic-rag");
  await expect(page.getByText("Workflow Evaluation Project")).toBeVisible();
  // Check that the run selector is visible
  await expect(
    page.getByText("Select workflow evaluation runs to compare..."),
  ).toBeVisible();
  // sleep for 500ms
  await page.waitForTimeout(500);
  // Click on the run selector
  await page.getByText("Select workflow evaluation runs to compare...").click();
  await page.waitForTimeout(1000);

  // Check that the run selector has the correct runs using CommandItem selectors
  await expect(
    page.getByRole("option").filter({ hasText: "aac7e7" }),
  ).toBeVisible();
  await expect(
    page.getByRole("option").filter({ hasText: "8fddbd" }),
  ).toBeVisible();
  await page.waitForTimeout(500);

  // Select 2 runs using CommandItem selectors
  await page.getByRole("option").filter({ hasText: "aac7e7" }).click();
  await page.waitForTimeout(500);
  await page.getByRole("option").filter({ hasText: "8fddbd" }).click();
  await page.waitForTimeout(500);
  // Click away from the run selector
  await page.click("body");
  // sleep for 500ms
  await page.waitForTimeout(500);
  // Check that the results table is visible
  await expect(page.getByText("Task Name")).toBeVisible();
  // Check that the results table has the correct columns
  await expect(page.getByText("judge_score")).toBeVisible();
  // Verify specific task IDs exist in the table
  await expect(
    page.getByText("0277e959b5a291f2aabd4a38dc831846903e6d74"),
  ).toBeVisible();

  await expect(
    page.getByText("agent-gpt-4.1-mini-compact_context-gemini-2.5-flash"),
  ).toBeVisible();

  await expect(
    page.getByText("agent-gemini-2.5-flash-compact_context-baseline"),
  ).toBeVisible();

  // Count rows to ensure we have data (expect more than just the header row)
  const rowCount = await page.locator("tr").count();
  expect(rowCount).toBeGreaterThan(1);
});
