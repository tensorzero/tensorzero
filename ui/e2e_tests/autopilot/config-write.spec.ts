import { test, expect } from "@playwright/test";
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

test.describe("Config writing", () => {
  // Clean up after each test by resetting git state
  test.afterEach(async () => {
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

    // 7. Click "Write All Configs" button
    const writeButton = page.getByRole("button", { name: "Write All Configs" });
    await expect(writeButton).toBeVisible({ timeout: 10000 });
    await writeButton.click();

    // 8. Wait for success toast and verify files were actually written
    const toastLocator = page
      .getByRole("status")
      .filter({ hasText: "Configs written" });
    await expect(toastLocator).toBeVisible({ timeout: 30000 });

    // 9. Verify the TOML config was updated
    const configPath = path.join(AUTOPILOT_CONFIG_DIR, "tensorzero.toml");
    const configContent = fs.readFileSync(configPath, "utf-8");
    const parsedToml = parseToml(configContent) as Record<string, unknown>;

    // Assert variant exists in extract_entities function
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

    // 10. Verify template file was created
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
});
