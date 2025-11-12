import { test, expect } from "@playwright/test";
import type { ProviderConfig } from "~/types/tensorzero";
import { formatProvider } from "~/utils/providers";

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

  [
    {
      provider: "OpenAI",
      model: "gpt-4o-2024-08-06",
      results: `
    [models.mock-inference-finetune-1234]
    routing = [ "mock-inference-finetune-1234" ]

    [models.mock-inference-finetune-1234.providers.mock-inference-finetune-1234]
    type = "openai"
    model_name = "mock-inference-finetune-1234"
    `,
    },
    {
      provider: "Fireworks",
      model: "llama-3.2-3b-instruct",
      results: `
[models."accounts/fake_fireworks_account/models/mock-fireworks-model"]
routing = [ "accounts/fake_fireworks_account/models/mock-fireworks-model" ]

[models."accounts/fake_fireworks_account/models/mock-fireworks-model".providers."accounts/fake_fireworks_account/models/mock-fireworks-model"]
type = "fireworks"
model_name = "accounts/fake_fireworks_account/models/mock-fireworks-model"
`,
    },
    {
      provider: "Together",
      model: "gpt-oss-20b",
      results: `
      type = "together"
      `,
      // Together mock SFT provider randomly generates model name so we'll just
      // assert that we have a model type together
    },
  ]
    .slice(2)
    .forEach(({ provider, model, results }) => {
      test(`@slow should fine-tune on filtered metric data with a mocked ${provider} server`, async ({
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
          .getByRole("option", { name: [model, provider].join(" ") })
          .click();
        await page
          .getByRole("button", { name: "Start Fine-tuning Job" })
          .click();

        await page
          .getByText("running", { exact: true })
          .waitFor({ timeout: 12000 });

        await expect(page.getByText(model)).toBeVisible();
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
        await expect(page.getByText(results)).toBeVisible();
      });
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

    await page.getByText("running", { exact: true }).waitFor({ timeout: 12000 });
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

  test("should show demonstration metric option for write_haiku function", async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");

    // Select write_haiku function
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    // Open metric selector
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a metric" })
      .click();

    // Verify demonstration option is visible and can be selected
    await expect(
      page.getByText("demonstration", { exact: true }),
    ).toBeVisible();
    await page.getByText("demonstration", { exact: true }).click();

    // Verify the metric selection is shown in the form
    await expect(
      page.getByRole("combobox").filter({ hasText: "demonstration" }),
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
      .waitFor({ timeout: 12000 });
  });
});

test.describe("should expose configured providers", () => {
  const providers: ProviderConfig["type"][] = [
    "openai",
    "fireworks",
    "gcp_vertex_gemini",
    "together",
  ];

  // ensure each provider we expect is in the list
  providers.forEach((provider) => {
    test(provider, async ({ page }) => {
      await page.goto("/optimization/supervised-fine-tuning");

      const modelName = "test-name";
      const providerName = formatProvider(provider).name;

      await page
        .getByRole("combobox")
        .filter({ hasText: "Select a model..." })
        .click();

      await page.getByPlaceholder("Search models...").fill(modelName);

      // Wait for the options to load
      await page
        .getByRole("option", {
          name: `${modelName} ${providerName}`,
        })
        .waitFor();
    });
  });

  // check that the number of providers is equal (this will ensure
  // that this test stays up to date)
  test(`should expose all ${providers.length} configured providers`, async ({
    page,
  }) => {
    await page.goto("/optimization/supervised-fine-tuning");

    const modelName = "test-name";

    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a model..." })
      .click();

    await page.getByPlaceholder("Search models...").fill(modelName);

    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("option")).toHaveCount(providers.length);
  });
});
