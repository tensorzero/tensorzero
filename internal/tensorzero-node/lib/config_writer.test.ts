import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { ConfigWriter } from "./index";
import type { EditPayload } from "./bindings";
import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";

describe("ConfigWriter", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "config-writer-test-"));
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("should create a ConfigWriter and apply an UpsertVariant edit", async () => {
    // Create a sample config file
    const configContent = `[functions.my_function]
type = "chat"

[functions.my_function.variants.baseline]
type = "chat_completion"
model = "gpt-4"
`;
    await fs.writeFile(path.join(tmpDir, "tensorzero.toml"), configContent);

    // Create ConfigWriter
    const globPattern = path.join(tmpDir, "**/*.toml");
    const writer = await ConfigWriter.new(globPattern);

    // Create an UpsertVariant edit
    const edit: EditPayload = {
      operation: "upsert_variant",
      function_name: "my_function",
      variant_name: "new_variant",
      variant: {
        timeouts: null,
        type: "chat_completion",
        weight: null,
        model: "gpt-4o",
        system_template: {
          __tensorzero_remapped_path: "inline",
          __data: "You are a helpful assistant.",
        },
        user_template: null,
        assistant_template: null,
        input_wrappers: null,
        templates: {},
        temperature: 0.7,
        top_p: null,
        max_tokens: 1000,
        presence_penalty: null,
        frequency_penalty: null,
        seed: null,
        stop_sequences: null,
        json_mode: null,
        retries: {
          num_retries: 3,
          max_delay_s: 10,
        },
      },
    };

    // Apply the edit
    const writtenPaths = await writer.applyEdit(edit);

    // Verify we got paths back
    expect(
      writtenPaths.length,
      "expected at least one file to be written",
    ).toBeGreaterThan(0);

    // Find the config file path (ends with .toml)
    const configFilePath = writtenPaths.find((p) => p.endsWith(".toml"));
    expect(
      configFilePath,
      "expected a .toml config file to be written",
    ).toBeDefined();

    // Verify the new variant is in the config
    const updatedConfig = await fs.readFile(configFilePath!, "utf-8");
    expect(updatedConfig).toContain("new_variant");
    expect(updatedConfig).toContain("gpt-4o");

    // Find the template file path
    const templateFilePath = writtenPaths.find((p) =>
      p.endsWith("system_template.minijinja"),
    );
    expect(
      templateFilePath,
      "expected a system_template.minijinja file to be written",
    ).toBeDefined();

    const templateContent = await fs.readFile(templateFilePath!, "utf-8");
    expect(templateContent).toBe("You are a helpful assistant.");
  });

  it("should fail with invalid glob pattern", async () => {
    const globPattern = path.join(tmpDir, "**/*.toml");
    // No config files exist, should fail
    await expect(ConfigWriter.new(globPattern)).rejects.toThrow();
  });

  it("should fail with invalid JSON", async () => {
    // Create a sample config file
    const configContent = `[functions.my_function]
type = "chat"
`;
    await fs.writeFile(path.join(tmpDir, "tensorzero.toml"), configContent);

    const globPattern = path.join(tmpDir, "**/*.toml");
    const writer = await ConfigWriter.new(globPattern);

    // Try to apply an invalid edit (access the native method directly via workaround)
    // Since applyEdit expects EditPayload, we need to test via the native binding
    // This tests that invalid JSON is handled properly
    const nativeWriter = (writer as unknown as { nativeConfigWriter: unknown })
      .nativeConfigWriter as { applyEdit: (json: string) => Promise<string[]> };
    await expect(nativeWriter.applyEdit("invalid json")).rejects.toThrow(
      "Failed to parse EditPayload",
    );
  });
});
