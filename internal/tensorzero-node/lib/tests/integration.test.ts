import { describe, it, expect } from "vitest";
import { TensorZeroClient } from "../index.js";

describe("TensorZeroClient Integration Tests", () => {
  it("should be able to import TensorZeroClient", () => {
    expect(TensorZeroClient).toBeDefined();
    expect(typeof TensorZeroClient).toBe("function");
  });

  it("client creation throws on a bad config path", async () => {
    // This should throw an error that contains "Failed to parse config: Internal TensorZero Error: Config file not found: "foo""
    expect(async () => await TensorZeroClient.build("foo")).rejects.toThrow(
      'Failed to parse config: Internal TensorZero Error: Config file not found: "foo"'
    );
  });

  it("should have required methods and initialize without credentials", async () => {
    const client = await buildClient();
    expect(typeof client.experimentalLaunchOptimizationWorkflow).toBe(
      "function"
    );
    expect(typeof client.experimentalPollOptimization).toBe("function");
  });

  it("should be able to get function config", async () => {
    const client = await buildClient();
    const extractEntitiesConfig =
      await client.getFunctionConfig("extract_entities");
    expect(extractEntitiesConfig).toBeDefined();
    expect(extractEntitiesConfig.variants).toBeDefined();
    const extractEntitiesVariantNames = Object.keys(
      extractEntitiesConfig.variants
    );
    expect(extractEntitiesVariantNames.length).toBe(6);
    for (const variantName of extractEntitiesVariantNames) {
      const variant = extractEntitiesConfig.variants[variantName];
      expect(variant).toBeDefined();
      expect(["chat_completion", "dicl"]).toContain(variant!.inner.type);
      if (variant!.inner.type === "chat_completion") {
        expect(variant!.inner.model).toBeDefined();
        expect(variant!.inner.system_template).toBeDefined();
        expect(variant!.inner.json_mode).toBeDefined();
        expect(variant!.inner.json_mode).toBe("strict");
      }
    }
    const generateSecretConfig =
      await client.getFunctionConfig("generate_secret");
    expect(generateSecretConfig).toBeDefined();
    expect(generateSecretConfig.type).toBe("json");
    if (generateSecretConfig.type === "json") {
      expect(generateSecretConfig.output_schema.value).toEqual({
        additionalProperties: false,
        properties: {
          secret: {
            type: "string",
          },
          thinking: {
            type: "string",
          },
        },
        required: ["thinking", "secret"],
        type: "object",
      });
    }
    expect(generateSecretConfig.variants).toBeDefined();
    const generateSecretVariantNames = Object.keys(
      generateSecretConfig.variants
    );
    expect(generateSecretVariantNames.length).toBe(1);
    const generateSecretVariant =
      generateSecretConfig.variants[generateSecretVariantNames[0]];
    expect(generateSecretVariant).toBeDefined();
    expect(generateSecretVariant!.inner.type).toBe("chat_completion");
  });

  it("should be able to get metric config", async () => {
    const client = await buildClient();
    const metricConfig = await client.getMetricConfig("exact_match");
    expect(metricConfig).toBeDefined();
    expect(metricConfig.type).toBe("boolean");
    expect(metricConfig.level).toBe("inference");
    expect(metricConfig.optimize).toBe("max");
  });
});

async function buildClient() {
  process.env.OPENAI_API_KEY = undefined;
  return await TensorZeroClient.build(
    "../../ui/fixtures/config/tensorzero.toml"
  );
}
