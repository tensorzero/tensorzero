import { test, expect } from "@playwright/test";

test("should show the evaluation result page", async ({ page }) => {
  await page.goto("/evaluations");
  await expect(page.getByText("New Run")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

// This test depends on model inference cache hits (within ClickHouse)
// If it starts failing, you may need to regenerate the model inference cache
test("push the new run button, launch an evaluation", async ({ page }) => {
  test.setTimeout(600_000);
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByText("Select an evaluation").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "entity_extraction" }).click();
  await page.waitForTimeout(500);
  await page.getByText("Select a dataset").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "foo" }).click();
  await page.waitForTimeout(500);
  await page.getByText("Select a variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "gpt4o_mini_initial_prompt" }).click();
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();
  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute("data-running", "true");
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute("data-running", "false", {timeout: 500_000});
  await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

// This test depends on model inference cache hits (within ClickHouse)
// If it starts failing, you may need to regenerate the model inference cache
test("push the new run button, launch an image evaluation", async ({
  page,
}) => {
  test.setTimeout(600_000);
  await page.goto("/evaluations");
  await page.waitForTimeout(500);
  await page.getByText("New Run").click();
  await page.waitForTimeout(500);
  await page.getByText("Select an evaluation").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "images" }).click();
  await page.waitForTimeout(500);
  await page.getByText("Select a dataset").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "baz" }).click();
  await page.waitForTimeout(500);
  await page.getByText("Select a variant").click();
  await page.waitForTimeout(500);
  await page.getByRole("option", { name: "honest_answer" }).click();
  await page.getByRole("button", { name: "Launch" }).click();

  await expect(
    page.getByText("Select evaluation runs to compare..."),
  ).toBeVisible();

  // Wait for evals to start, then wait for them to finish
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute("data-running", "true");
  await expect(page.getByTestId("auto-refresh-wrapper")).toHaveAttribute("data-running", "false", {timeout: 500_000});

  await expect(page.getByText("matches_reference")).toBeVisible();
  await expect(page.getByText("n=", { exact: false }).first()).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});
