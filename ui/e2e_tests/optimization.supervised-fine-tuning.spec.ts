import { test, expect } from "@playwright/test";

test("should show the supervised fine-tuning page", async ({ page }) => {
  await page.goto("/optimization/supervised-fine-tuning");
  await expect(page.getByText("Advanced Parameters")).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

test.describe("Custom user agent", () => {
  // We look for this user agent in the fine-tuning code, and configure a
  // shorter polling interval. This avoids the need to wait 10 seconds in
  // between polling mock-inference-provider
  test.use({ userAgent: "TensorZeroE2E" });

  test("@slow should fine-tune on filtered metric data with a mocked OpenAI server", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();
    await page.getByText("exact_match", { exact: true }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a variant name" })
      .click();
    await page
      .getByLabel("gpt4o_mini_initial_prompt")
      .getByText("gpt4o_mini_initial_prompt")
      .click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();
    await page
      .getByRole("option", { name: "gpt-4o-2024-08-06 OpenAI" })
      .click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();

    await page.getByText("running", { exact: true }).waitFor({ timeout: 3000 });

    await expect(page.getByText("gpt-4o-2024-08-06")).toBeVisible();
    await expect(
      page.getByRole("link", { name: "extract_entities" }),
    ).toBeVisible();
    await expect(page.getByText("exact_match")).toBeVisible();
    await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

    // The mock server completes the job after 2 seconds (so that we can eventually test the progress bar),
    // so wait for 3 seconds here to make sure it completes
    await page
      .getByText("completed", { exact: true })
      .waitFor({ timeout: 3000 });
    await expect(
      page.getByText(`
[models.mock-inference-finetune-1234]
routing = [ "mock-inference-finetune-1234" ]

[models.mock-inference-finetune-1234.providers.mock-inference-finetune-1234]
type = "openai"
model_name = "mock-inference-finetune-1234"
`),
    ).toBeVisible();
  });

  test("@slow should fine-tune on demonstration data with a mocked OpenAI server", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();
    await page.getByText("demonstration", { exact: true }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a variant name" })
      .click();
    await page
      .getByLabel("gpt4o_mini_initial_prompt")
      .getByText("gpt4o_mini_initial_prompt")
      .click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();
    await page
      .getByRole("option", { name: "gpt-4o-2024-08-06 OpenAI" })
      .click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();

    await page.getByText("running", { exact: true }).waitFor({ timeout: 3000 });
    await expect(page.getByText("gpt-4o-2024-08-06")).toBeVisible();
    await expect(
      page.getByRole("link", { name: "extract_entities" }),
    ).toBeVisible();
    await expect(page.getByText("demonstration")).toBeVisible();
    await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

    // The mock server completes the job after 2 seconds (so that we can eventually test the progress bar),
    // so wait for 3 seconds here to make sure it completes
    await page
      .getByText("completed", { exact: true })
      .waitFor({ timeout: 3000 });

    await expect(
      page.getByText(`
[models.mock-inference-finetune-1234]
routing = [ "mock-inference-finetune-1234" ]

[models.mock-inference-finetune-1234.providers.mock-inference-finetune-1234]
type = "openai"
model_name = "mock-inference-finetune-1234"
`),
    ).toBeVisible();
  });

  test("@slow should fine-tune on image data with a mocked OpenAI server", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "image_judger" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();
    await page.getByRole("option", { name: "None" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a variant name" })
      .click();
    await page.getByLabel("honest_answer").getByText("honest_answer").click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();
    await page
      .getByRole("option", { name: "gpt-4o-2024-08-06 OpenAI" })
      .click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();

    await page
      .getByText("running", { exact: true })
      .waitFor({ timeout: 60000 });
    await expect(page.getByText("gpt-4o-2024-08-06")).toBeVisible();
    await expect(
      page.getByRole("link", { name: "image_judger" }),
    ).toBeVisible();
    await expect(page.getByText("None")).toBeVisible();
    await expect(page.getByText("honest_answer")).toBeVisible();

    // The mock server completes the job after 2 seconds (so that we can eventually test the progress bar),
    // so wait for 3 seconds here to make sure it completes
    await page
      .getByText("completed", { exact: true })
      .waitFor({ timeout: 3000 });
    await expect(
      page.getByText(`
[models.mock-inference-finetune-1234]
routing = [ "mock-inference-finetune-1234" ]

[models.mock-inference-finetune-1234.providers.mock-inference-finetune-1234]
type = "openai"
model_name = "mock-inference-finetune-1234"
`),
    ).toBeVisible();
  });

  test("@slow should fine-tune with a mocked Fireworks server", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();
    await page.getByText("exact_match", { exact: true }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a variant name" })
      .click();
    await page
      .getByLabel("gpt4o_mini_initial_prompt")
      .getByText("gpt4o_mini_initial_prompt")
      .click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();
    await page
      .getByRole("option", { name: "llama-3.2-3b-instruct Fireworks" })
      .click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();

    await page.getByText("running", { exact: true }).waitFor({ timeout: 3000 });
    await expect(
      page.getByText("accounts/fireworks/models/llama-v3p2-3b-instruct"),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "extract_entities" }),
    ).toBeVisible();
    await expect(page.getByText("exact_match")).toBeVisible();
    await expect(page.getByText("gpt4o_mini_initial_prompt")).toBeVisible();

    await expect(
      page.getByText(`
[models."accounts/fake_fireworks_account/models/mock-fireworks-model"]
routing = [ "accounts/fake_fireworks_account/models/mock-fireworks-model" ]

[models."accounts/fake_fireworks_account/models/mock-fireworks-model".providers."accounts/fake_fireworks_account/models/mock-fireworks-model"]
type = "fireworks"
model_name = "accounts/fake_fireworks_account/models/mock-fireworks-model"
`),
    ).toBeVisible();
  });
});

test.describe("Error handling", () => {
  test("should show an error when the model is an error model", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities" }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();
    await page.getByText("exact_match", { exact: true }).click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a variant name" })
      .click();
    await page
      .getByLabel("gpt4o_mini_initial_prompt")
      .getByText("gpt4o_mini_initial_prompt")
      .click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();
    // Fill in the input of the popover with "error"
    await page.getByPlaceholder("Search models...").fill("error");
    // Wait for the options to load
    await page.getByRole("option", { name: "error OpenAI" }).waitFor();
    // Click on the option that has text "error" and provider "OpenAI"
    await page.getByRole("option", { name: "error OpenAI" }).click();
    // Click on the Start Fine-tuning Job button
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();

    await page
      .getByText("failed because the model is an error model")
      .waitFor({ timeout: 3000 });
  });
});
