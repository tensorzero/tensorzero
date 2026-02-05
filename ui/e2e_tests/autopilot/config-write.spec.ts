import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { parse as parseToml } from "smol-toml";
import { deterministicTestAndAttempt } from "./helpers";

// Path to the autopilot repo config directory (from host perspective)
const AUTOPILOT_CONFIG_DIR = path.join(
  process.env.HOME || "",
  "autopilot/e2e_tests/fixtures/config",
);

/**
 * Clicks "Write All Configs" and waits for the API response to succeed.
 * Returns the parsed TOML config after writing.
 */
async function writeAllConfigsAndParse(
  page: Page,
): Promise<Record<string, unknown>> {
  const writeButton = page.getByRole("button", { name: "Write All Configs" });
  await expect(writeButton).toBeVisible({ timeout: 10000 });

  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("config-writes/write-all"),
    { timeout: 60000 },
  );
  await writeButton.click();
  const apiResponse = await responsePromise;
  // Status 200 = success, 500 = write failure (error toast would say "Failed to write configs")
  expect(apiResponse.status(), "Write All Configs API should succeed").toBe(
    200,
  );

  // Brief pause to ensure filesystem writes are visible to this process
  await page.waitForTimeout(500);

  const configPath = path.join(AUTOPILOT_CONFIG_DIR, "tensorzero.toml");
  const configContent = fs.readFileSync(configPath, "utf-8");
  return parseToml(configContent) as Record<string, unknown>;
}

test.describe("Config writing", () => {
  // Tests must run serially because they all write to the same config file.
  // Parallel execution causes ConfigWriter in one test to overwrite another's changes.
  test.describe.configure({ mode: "serial" });

  // Clean up after all tests by resetting git state
  test.afterAll(async () => {
    try {
      execSync(`git checkout -- e2e_tests/fixtures/config`, {
        cwd: path.join(process.env.HOME || "", "autopilot"),
      });
      execSync(`git clean -fd e2e_tests/fixtures/config`, {
        cwd: path.join(process.env.HOME || "", "autopilot"),
      });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to clean up config directory:", e);
    }
  });

  test("should write a new variant with template to config files", async ({
    page,
  }, testInfo) => {
    test.setTimeout(180000); // 3 minutes for LLM + file operations

    const variantName = deterministicTestAndAttempt(testInfo, "variant");

    // Build the variant JSON with the correct ResolvedTomlPathData format
    // for system_template so the set_variant tool accepts it.
    // The path is arbitrary since the config writing crate canonicalizes paths.
    const templateContent =
      "Extract all named entities (people, organizations, locations) from the given text and return them as JSON.";
    const variantJson = JSON.stringify({
      type: "chat_completion",
      model: "gpt-4o-mini-2024-07-18",
      system_template: {
        __tensorzero_remapped_path: "dummy",
        __data: templateContent,
      },
      temperature: 0,
      json_mode: "strict",
    });

    // 1. Navigate to new session
    await page.goto("/autopilot/sessions/new");
    await page.waitForLoadState("networkidle");

    // 2. Enable YOLO mode for auto-approval
    const yoloLabel = page.locator("label").filter({ hasText: "YOLO Mode" });
    await yoloLabel.click();
    await expect(yoloLabel.getByRole("switch")).toHaveAttribute(
      "data-state",
      "checked",
    );

    // 3. Send message with explicit tool call parameters (matching Rust e2e test pattern)
    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      `Call the set_variant tool to add a new variant to the extract_entities function.\n` +
        `Use these exact parameters:\n` +
        `- function_name: "extract_entities"\n` +
        `- variant_name: "${variantName}"\n` +
        `- variant: ${variantJson}\n\n` +
        `Only make this single tool call, nothing else.`,
    );
    await page.getByRole("button", { name: "Send message" }).click();

    // 4. Wait for redirect to session page
    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // 5. Wait for session to become idle (Ready status)
    await expect(page.getByText("Ready", { exact: true })).toBeVisible({
      timeout: 120000,
    });

    // 6. Verify no errors occurred
    await expect(page.getByText("Something went wrong")).not.toBeVisible();

    // 7. Write configs and parse TOML
    const parsedToml = await writeAllConfigsAndParse(page);

    // 8. Assert variant exists in extract_entities function
    const functionsDefs = parsedToml["functions"] as Record<string, unknown>;
    expect(functionsDefs, "Config should have functions section").toBeDefined();

    const extractEntities = functionsDefs["extract_entities"] as Record<
      string,
      unknown
    >;
    expect(
      extractEntities,
      "Config should have extract_entities function",
    ).toBeDefined();

    const variants = extractEntities.variants as Record<string, unknown>;
    expect(
      variants,
      "extract_entities should have variants section",
    ).toBeDefined();
    expect(
      variants[variantName],
      `Variant ${variantName} should exist`,
    ).toBeDefined();

    const variant = variants[variantName] as Record<string, unknown>;
    expect(variant.type, "Variant type should be chat_completion").toBe(
      "chat_completion",
    );
    expect(
      variant.model,
      "Variant model should be gpt-4o-mini-2024-07-18",
    ).toBe("gpt-4o-mini-2024-07-18");

    // 9. Verify template file was created
    // The config writer canonicalizes paths, so read it from the TOML.
    const writtenSystemTemplate = variant.system_template as string | undefined;
    expect(
      writtenSystemTemplate,
      "Variant should have a system_template path",
    ).toBeDefined();

    const fullTemplatePath = path.join(
      AUTOPILOT_CONFIG_DIR,
      writtenSystemTemplate!,
    );
    expect(
      fs.existsSync(fullTemplatePath),
      `Template file should exist at ${fullTemplatePath}`,
    ).toBe(true);

    const writtenTemplateContent = fs.readFileSync(fullTemplatePath, "utf-8");
    expect(
      writtenTemplateContent.length,
      "Template content should not be empty",
    ).toBeGreaterThan(0);
  });

  test("should write a new evaluation to config files", async ({
    page,
  }, testInfo) => {
    test.setTimeout(180000);

    const evaluationName = deterministicTestAndAttempt(testInfo, "evaluation");

    const evaluationJson = JSON.stringify({
      type: "inference",
      function_name: "write_haiku",
      evaluators: {},
    });

    // 1. Navigate to new session
    await page.goto("/autopilot/sessions/new");
    await page.waitForLoadState("networkidle");

    // 2. Enable YOLO mode
    const yoloLabel = page.locator("label").filter({ hasText: "YOLO Mode" });
    await yoloLabel.click();
    await expect(yoloLabel.getByRole("switch")).toHaveAttribute(
      "data-state",
      "checked",
    );

    // 3. Send message with explicit tool call parameters
    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      `Call the set_evaluation tool to create a new evaluation.\n` +
        `Use these exact parameters:\n` +
        `- evaluation_name: "${evaluationName}"\n` +
        `- evaluation: ${evaluationJson}\n\n` +
        `Only make this single tool call, nothing else.`,
    );
    await page.getByRole("button", { name: "Send message" }).click();

    // 4. Wait for redirect to session page
    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // 5. Wait for session to become idle (Ready status)
    await expect(page.getByText("Ready", { exact: true })).toBeVisible({
      timeout: 120000,
    });

    // 6. Verify no errors occurred
    await expect(page.getByText("Something went wrong")).not.toBeVisible();

    // 7. Write configs and parse TOML
    const parsedToml = await writeAllConfigsAndParse(page);

    // 8. Assert evaluation exists
    const evaluations = parsedToml["evaluations"] as Record<string, unknown>;
    expect(evaluations, "Config should have evaluations section").toBeDefined();

    const evaluation = evaluations[evaluationName] as Record<string, unknown>;
    expect(
      evaluation,
      `Evaluation ${evaluationName} should exist`,
    ).toBeDefined();

    expect(evaluation.type, "Evaluation type should be inference").toBe(
      "inference",
    );
    expect(
      evaluation.function_name,
      "Evaluation function_name should be write_haiku",
    ).toBe("write_haiku");
  });

  test("should write a new llm_judge evaluator with system_instructions to config files", async ({
    page,
  }, testInfo) => {
    test.setTimeout(180000);

    const evaluatorName = deterministicTestAndAttempt(testInfo, "evaluator");

    const instructionsContent =
      "Judge whether the haiku follows the 5-7-5 syllable pattern. Return true if it does, false otherwise.";
    const evaluatorJson = JSON.stringify({
      type: "llm_judge",
      output_type: "boolean",
      optimize: "max",
      variants: {
        mini: {
          type: "chat_completion",
          model: "gpt-4o-mini-2024-07-18",
          json_mode: "off",
          system_instructions: {
            __tensorzero_remapped_path: "dummy",
            __data: instructionsContent,
          },
        },
      },
    });

    // 1. Navigate to new session
    await page.goto("/autopilot/sessions/new");
    await page.waitForLoadState("networkidle");

    // 2. Enable YOLO mode
    const yoloLabel = page.locator("label").filter({ hasText: "YOLO Mode" });
    await yoloLabel.click();
    await expect(yoloLabel.getByRole("switch")).toHaveAttribute(
      "data-state",
      "checked",
    );

    // 3. Send message with explicit tool call parameters
    const messageInput = page.getByRole("textbox");
    await messageInput.fill(
      `Call the set_evaluator tool to add an evaluator to the haiku evaluation.\n` +
        `Use these exact parameters:\n` +
        `- evaluation_name: "haiku"\n` +
        `- evaluator_name: "${evaluatorName}"\n` +
        `- evaluator: ${evaluatorJson}\n\n` +
        `Only make this single tool call, nothing else.`,
    );
    await page.getByRole("button", { name: "Send message" }).click();

    // 4. Wait for redirect to session page
    await expect(page).toHaveURL(/\/autopilot\/sessions\/[a-f0-9-]+$/, {
      timeout: 30000,
    });

    // 5. Wait for session to become idle (Ready status)
    await expect(page.getByText("Ready", { exact: true })).toBeVisible({
      timeout: 120000,
    });

    // 6. Verify no errors occurred
    await expect(page.getByText("Something went wrong")).not.toBeVisible();

    // 7. Write configs and parse TOML
    const parsedToml = await writeAllConfigsAndParse(page);

    // 8. Assert evaluator exists under haiku evaluation
    const evaluations = parsedToml["evaluations"] as Record<string, unknown>;
    expect(evaluations, "Config should have evaluations section").toBeDefined();

    const haikuEval = evaluations["haiku"] as Record<string, unknown>;
    expect(haikuEval, "Config should have haiku evaluation").toBeDefined();

    const evaluators = haikuEval.evaluators as Record<string, unknown>;
    expect(
      evaluators,
      "haiku evaluation should have evaluators section",
    ).toBeDefined();

    const evaluator = evaluators[evaluatorName] as Record<string, unknown>;
    expect(evaluator, `Evaluator ${evaluatorName} should exist`).toBeDefined();

    expect(evaluator.type, "Evaluator type should be llm_judge").toBe(
      "llm_judge",
    );
    expect(
      evaluator.output_type,
      "Evaluator output_type should be boolean",
    ).toBe("boolean");
    expect(evaluator.optimize, "Evaluator optimize should be max").toBe("max");

    // Verify variant exists
    const variants = evaluator.variants as Record<string, unknown>;
    expect(variants, "Evaluator should have variants section").toBeDefined();

    const miniVariant = variants["mini"] as Record<string, unknown>;
    expect(miniVariant, "Variant mini should exist").toBeDefined();
    expect(
      miniVariant.model,
      "Variant model should be gpt-4o-mini-2024-07-18",
    ).toBe("gpt-4o-mini-2024-07-18");

    // Verify system_instructions file was created
    const writtenInstructions = miniVariant.system_instructions as
      | string
      | undefined;
    expect(
      writtenInstructions,
      "Variant should have a system_instructions path",
    ).toBeDefined();

    const fullInstructionsPath = path.join(
      AUTOPILOT_CONFIG_DIR,
      writtenInstructions!,
    );
    expect(
      fs.existsSync(fullInstructionsPath),
      `Instructions file should exist at ${fullInstructionsPath}`,
    ).toBe(true);

    const writtenInstructionsContent = fs.readFileSync(
      fullInstructionsPath,
      "utf-8",
    );
    expect(
      writtenInstructionsContent.length,
      "Instructions content should not be empty",
    ).toBeGreaterThan(0);
  });
});
