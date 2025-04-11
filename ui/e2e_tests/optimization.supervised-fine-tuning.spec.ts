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
  test.use({ userAgent: 'TensorZeroE2E' });


  test("should fine-tune with a mocked OpenAI server", async ({ page }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities JSON" }).click();
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
    await page.getByRole("option", { name: "gpt-4o-2024-08-06 OpenAI" }).click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();
    // Expect redirect
    await page.waitForURL("/optimization/supervised-fine-tuning/*?backend=*");

    let regex;
    if (process.env.TENSORZERO_UI_FF_ENABLE_PYTHON === "1") {
      regex = /\?backend=python$/;
    } else {
      regex = /\?backend=nodejs$/;
    }

    // Verify that we used the correct fine-tuning backend
    expect(page.url()).toEqual(expect.stringMatching(regex));

    await page.getByText("running", { exact: true }).waitFor({ timeout: 3000 });
    await expect(page.locator("body")).toContainText(
      "Base Model: gpt-4o-2024-08-06",
    );
    await expect(page.locator("body")).toContainText(
      "Function: extract_entities",
    );
    await expect(page.locator("body")).toContainText("Metric: exact_match");
    await expect(page.locator("body")).toContainText(
      "Prompt: gpt4o_mini_initial_prompt",
    );

    // The mock server completes the job after 2 seconds (so that we can eventually test the progress bar),
    // so wait for 3 seconds here to make sure it completes
    await page
      .getByText("completed", { exact: true })
      .waitFor({ timeout: 3000 })
    await expect(page.locator("body")).toContainText("Configuration");
    await expect(page.locator("body")).toContainText(`
[models.mock-inference-finetune-1234]
routing = [ "mock-inference-finetune-1234" ]

[models.mock-inference-finetune-1234.providers.mock-inference-finetune-1234]
type = "openai"
model_name = "mock-inference-finetune-1234"
`);
  });

  test("should fine-tune with a mocked Fireworks server", async ({ page }) => {
    await page.goto("/optimization/supervised-fine-tuning");
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select a function" })
      .click();
    await page.getByRole("option", { name: "extract_entities JSON" }).click();
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
    await page.getByRole("option", { name: "llama-3.2-3b-instruct Fireworks" }).click();
    await page.getByRole("button", { name: "Start Fine-tuning Job" }).click();
    // Expect redirect
    await page.waitForURL("/optimization/supervised-fine-tuning/*?backend=*");

    let regex;
    if (process.env.TENSORZERO_UI_FF_ENABLE_PYTHON === "1") {
      regex = /\?backend=python$/;
    } else {
      regex = /\?backend=nodejs$/;
    }

    // Verify that we used the correct fine-tuning backend
    expect(page.url()).toEqual(expect.stringMatching(regex));

    await page.getByText("running", { exact: true }).waitFor({ timeout: 3000 });
    await expect(page.locator("body")).toContainText(
      "Base Model: accounts/fireworks/models/llama-v3p2-3b-instruct",
    );
    await expect(page.locator("body")).toContainText(
      "Function: extract_entities",
    );
    await expect(page.locator("body")).toContainText("Metric: exact_match");
    await expect(page.locator("body")).toContainText(
      "Prompt: gpt4o_mini_initial_prompt",
    );

    await page
      .getByText("completed", { exact: true })
    await expect(page.locator("body")).toContainText("Configuration");
    await expect(page.locator("body")).toContainText(`
[models."accounts/fake_fireworks_account/models/mock-fireworks-model"]
routing = [ "accounts/fake_fireworks_account/models/mock-fireworks-model" ]

[models."accounts/fake_fireworks_account/models/mock-fireworks-model".providers."accounts/fake_fireworks_account/models/mock-fireworks-model"]
type = "fireworks"
model_name = "accounts/fake_fireworks_account/models/mock-fireworks-model"
`);
  });
})