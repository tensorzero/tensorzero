import { test, expect } from "@playwright/test";

test.describe("Custom Named Templates and Schemas", () => {
  test("should display custom-named templates on function page", async ({
    page,
  }) => {
    await page.goto("/observability/functions/custom_template_test");

    // Check that the function page loads (use heading instead of text to avoid ambiguity)
    await expect(page.getByRole("heading", { name: "Variants" })).toBeVisible();

    // Check that custom template names appear (exactly as in config)
    await expect(page.getByText("greeting_template")).toBeVisible();
    await expect(page.getByText("analysis_prompt")).toBeVisible();
    await expect(page.getByText("fun_fact_topic")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should display custom-named schemas on function page", async ({
    page,
  }) => {
    await page.goto("/observability/functions/custom_template_test");

    // Check that custom schema names appear (exactly as in config)
    await expect(page.getByText("greeting_template")).toBeVisible();
    await expect(page.getByText("analysis_prompt")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should display custom-named templates on variant page", async ({
    page,
  }) => {
    await page.goto(
      "/observability/functions/custom_template_test/variants/baseline",
    );

    // Template tabs should be visible (exactly as in config)
    await expect(page.getByText("greeting_template")).toBeVisible();
    await expect(page.getByText("analysis_prompt")).toBeVisible();
    await expect(page.getByText("fun_fact_topic")).toBeVisible();

    // Click on different template tabs to verify they work
    await page.getByText("analysis_prompt").click();
    await expect(page.getByText("Analyze the following data")).toBeVisible();

    await page.getByText("fun_fact_topic").click();
    await expect(page.getByText("Share a fun fact about")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should still support legacy system/user/assistant templates", async ({
    page,
  }) => {
    // Test a function using legacy template names
    await page.goto(
      "/observability/functions/extract_entities/variants/gpt4o_mini_initial_prompt",
    );

    // Legacy system template should still be displayed (lowercase as in config)
    await expect(page.getByText("system")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should still support legacy system/user schemas", async ({ page }) => {
    // Test a function using legacy schema names
    await page.goto("/observability/functions/judge_answer");

    // Legacy system and user schemas should still be displayed (lowercase as in config)
    await expect(page.getByText("system")).toBeVisible();
    await expect(page.getByText("user")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });

  test("should show empty state when no templates exist", async ({ page }) => {
    // Navigate to a function/variant with no templates
    await page.goto(
      "/observability/functions/image_judger/variants/honest_answer",
    );

    // Should show empty state message
    await expect(page.getByText("No templates defined")).toBeVisible();

    // Assert no errors
    await expect(page.getByText("error", { exact: false })).not.toBeVisible();
  });
});
