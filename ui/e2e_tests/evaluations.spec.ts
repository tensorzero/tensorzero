import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto("/evaluations");
  await expect(page.getByText("New Run")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("push the new run button, launch an evaluation", async ({ page }) => {
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
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();
  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "true",
  );
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "false",
    { timeout: 500_000 },
  );
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("push the new run button, launch an image evaluation", async ({
  page,
}) => {
  test.setTimeout(600_000);
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select evaluation").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "images" }).click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select dataset").click();
  await page.waitForTimeout(500);
  await page.locator('[data-dataset-name="baz"]').click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "honest_answer" }).click();
  // IMPORTANT - we need to set concurrency to 1 in order to prevent a race condition
  // when regenerating fixtures, as we intentionally have multiple datapoints with
  // identical inputs. See https://www.notion.so/tensorzerodotcom/Evaluations-cache-non-determinism-23a7520bbad3801f80fceaa7e859ce06
  await page.getByTestId("concurrency-limit").fill("1");
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();

  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "true",
  );
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "false",
    { timeout: 500_000 },
  );

  await expect(page.getByText("matches_reference")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test("run evaluation with dataset with no output", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select evaluation").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "haiku" }).click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select dataset").click();
  await page.waitForTimeout(500);
  await page.locator('[data-dataset-name="no_output"]').click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "initial_prompt_gpt4o_mini" }).click();
  await page.getByTestId("concurrency-limit").fill("5");
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();
  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "true",
  );
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "false",
    { timeout: 500_000 },
  );
  await expect(page.getByText("initial_prompt_gpt4o_mini")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

// Uses dummy::slow (5s per inference) to guarantee the evaluation is still
// running when we click Stop. No model inference cache needed.
const CANCEL_TEST_DATASET_SIZE = 10;
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
  await page.getByRole("option", { name: "dummy_evaluation" }).click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select dataset").click();
  await page.waitForTimeout(500);
  await page.locator('[data-dataset-name="cancel_test"]').click();
  await page.waitForTimeout(500);
  await page.getByPlaceholder("Select variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "dummy_slow" }).click();
  // Concurrency 1 ensures sequential processing so each datapoint takes ~5s
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

  // Wait for at least one datapoint to complete (~5s per datapoint with dummy::slow)
  await page.waitForTimeout(8_000);

  // Click the Stop button
  const stopButton = page.getByRole("button", { name: "Stop" });
  await expect(stopButton).toBeVisible();
  await stopButton.click();

  // Verify the button shows the "Stopping..." state
  await expect(page.getByText("Stopping...")).toBeVisible();

  // Wait for the evaluation to actually stop
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
  expect(
    n,
    "Should have fewer than all datapoints (cancellation stopped the evaluation)",
  ).toBeLessThan(CANCEL_TEST_DATASET_SIZE);
});

// This test verifies that adaptive stopping parameters work correctly
test("launch evaluation with adaptive stopping parameters", async ({
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
  await page.getByTestId("concurrency-limit").fill("1");

  // Fill in max_datapoints parameter
  await page.locator("#max_datapoints").fill("10");

  // Open Advanced Parameters accordion
  await page.getByText("Advanced Parameters").click();
  await page.waitForTimeout(500);

  // Fill in adaptive stopping parameters
  await page.locator("#precision_target_exact_match").fill("0.5");

  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();

  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "true",
  );
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute(
    "data-running",
    "false",
    { timeout: 500_000 },
  );

  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

  // Verify the evaluation ran with at most max_datapoints (10)
  const statsText = await page
    .getByText("n=", { exact: false })
    .first()
    .textContent();
  expect(statsText).toBeTruthy();
  const match = statsText?.match(/n=(\d+)/);
  expect(match).toBeTruthy();
  const n = parseInt(match![1], 10);
  expect(n).toBeLessThanOrEqual(10);

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
