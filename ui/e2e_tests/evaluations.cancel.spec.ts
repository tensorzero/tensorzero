import { test, expect } from "@playwright/test";

// This test depends on model inference cache hits (within ClickHouse)
// If it starts failing, you may need to regenerate the model inference cache
test("cancel a running evaluation and verify partial results", async ({
  page,
}) => {
  test.setTimeout(600_000);
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select evaluation").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "entity_extraction" }).click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select dataset").click();
  await page.waitForTimeout(500);
  await page.locator('[data-dataset-name="foo"]').click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "gpt4o_mini_initial_prompt" }).click();
  // Concurrency 1 ensures sequential processing, giving us time to cancel
  await page.getByTestId("concurrency-limit").fill("1");
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();

  // Wait for the evaluation to start running
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "true",
  );

  // Click the Stop button
  const stopButton = page.getByRole("button", { name: "Stop" });
  await expect(stopButton).toBeVisible();
  await stopButton.click();

  // Verify the button shows the "Stopping..." state
  await expect(page.getByText("Stopping...")).toBeVisible();

  // Wait for the evaluation to actually stop — 30s timeout is much shorter
  // than the 500s needed for natural completion, proving cancellation worked
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "false",
    { timeout: 30_000 },
  );

  // Verify partial results exist in ClickHouse
  const statsText = await page
    .getByText("n=", { exact: false })
    .first()
    .textContent();
  expect(
    statsText,
    "Should have some evaluation results in ClickHouse",
  ).toBeTruthy();
  const match = statsText?.match(/n=(\d+)/);
  expect(match, "Should match n=X pattern").toBeTruthy();
  const n = parseInt(match![1], 10);
  expect(n, "Should have at least 1 completed datapoint").toBeGreaterThan(0);
});
