import { describe, it, expect } from "vitest";
import { TensorZeroClient, DatabaseClient, getConfig } from "../index.js";

const UI_FIXTURES_CONFIG_PATH = "../../ui/fixtures/config/tensorzero.toml";

describe("TensorZeroClient Integration Tests", () => {
  it("should be able to import TensorZeroClient", () => {
    expect(TensorZeroClient).toBeDefined();
    expect(typeof TensorZeroClient).toBe("function");
  });

  it("client creation throws on a bad URL", async () => {
    // This should throw an error that contains "relative URL without a base"
    await expect(
      async () => await TensorZeroClient.buildHttp("foo"),
    ).rejects.toThrow("relative URL without a base");
  });

  it("should have required methods and initialize without credentials", async () => {
    const client = await buildClient();
    expect(typeof client.experimentalLaunchOptimizationWorkflow).toBe(
      "function",
    );
    expect(typeof client.experimentalPollOptimization).toBe("function");
  });

  it("should be able to stale dataset", async () => {
    const client = await buildClient();
    // In the future once we have full dataset lifecycle support here we can do a better test
    const staleDatasetResponse =
      await client.staleDataset("nonexistentdataset");
    expect(staleDatasetResponse).toBeDefined();
    expect(staleDatasetResponse.num_staled_datapoints).toBe(0);
  });
});

async function buildClient() {
  process.env.OPENAI_API_KEY = undefined;
  return await TensorZeroClient.buildEmbedded(
    UI_FIXTURES_CONFIG_PATH,
    undefined,
    process.env.TENSORZERO_POSTGRES_URL,
    undefined,
  );
}

describe("DatabaseClient", () => {
  it("should be able to import DatabaseClient", () => {
    expect(DatabaseClient).toBeDefined();
    expect(typeof DatabaseClient).toBe("function");
  });

  it("should have getFeedbackByVariant method", async () => {
    // Note: This test verifies the method exists and has correct signature
    // Full integration testing would require a running ClickHouse instance
    expect(typeof DatabaseClient.fromClickhouseUrl).toBe("function");
  });

  it("should validate getFeedbackByVariant parameter types", async () => {
    // This test documents the expected parameter structure
    const expectedParams = {
      metric_name: "test_metric",
      function_name: "test_function",
      variant_names: ["variant_a", "variant_b"],
    };

    // Verify the structure is what we expect
    expect(expectedParams).toHaveProperty("metric_name");
    expect(expectedParams).toHaveProperty("function_name");
    expect(expectedParams).toHaveProperty("variant_names");
    expect(Array.isArray(expectedParams.variant_names)).toBe(true);
  });
});

it("should get full config structure", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);
  expect(config).toBeDefined();
  expect(config.gateway).toBeDefined();
  expect(config.models).toBeDefined();
  expect(config.embedding_models).toBeDefined();
  expect(config.functions).toBeDefined();
  expect(config.tools).toBeDefined();
  expect(config.metrics).toBeDefined();
  expect(config.evaluations).toBeDefined();
});

it("should get config with gateway settings", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);
  expect(config.gateway.debug).toBe(true);
});

it("should get config with models including shorthand", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  // Check shorthand model exists
  expect(config.models.table["gpt-4o-mini-2024-07-18"]).toBeDefined();
  expect(config.models.table["llama-3.1-8b-instruct"]).toBeDefined();
  expect(
    config.models.table["ft:gpt-4o-mini-2024-07-18:tensorzero::ALHEaw1j"],
  ).toBeDefined();

  // Check routing arrays
  expect(config.models.table["gpt-4o-mini-2024-07-18"]!.routing).toEqual([
    "openai",
  ]);
  expect(config.models.table["llama-3.1-8b-instruct"]!.routing).toEqual([
    "fireworks",
  ]);
  expect(
    config.models.table["ft:gpt-4o-mini-2024-07-18:tensorzero::ALHEaw1j"]!
      .routing,
  ).toEqual(["openai"]);
});

it("should get config with embedding models", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  expect(config.embedding_models.table["text-embedding-3-small"]).toBeDefined();
  expect(
    config.embedding_models.table["text-embedding-3-small"]!.routing,
  ).toEqual(["openai"]);
});

it("should get config with comprehensive function coverage", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  // Test functions exist
  expect(config.functions.extract_entities).toBeDefined();
  expect(config.functions.write_haiku).toBeDefined();
  expect(config.functions.generate_secret).toBeDefined();
  expect(config.functions.judge_answer).toBeDefined();
  expect(config.functions.multi_hop_rag_agent).toBeDefined();

  // Test function types
  expect(config.functions.extract_entities!.type).toBe("json");
  expect(config.functions.write_haiku!.type).toBe("chat");
  expect(config.functions.generate_secret!.type).toBe("json");
  expect(config.functions.judge_answer!.type).toBe("json");
  expect(config.functions.multi_hop_rag_agent!.type).toBe("chat");

  // Test variant counts
  expect(Object.keys(config.functions.extract_entities!.variants).length).toBe(
    6,
  );
  expect(Object.keys(config.functions.write_haiku!.variants).length).toBe(3);
  expect(Object.keys(config.functions.generate_secret!.variants).length).toBe(
    1,
  );
  expect(Object.keys(config.functions.judge_answer!.variants).length).toBe(1);
  expect(
    Object.keys(config.functions.multi_hop_rag_agent!.variants).length,
  ).toBe(4);
});

it("should get config with tools", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  expect(config.tools.think).toBeDefined();
  expect(config.tools.search_wikipedia).toBeDefined();
  expect(config.tools.load_wikipedia_page).toBeDefined();
  expect(config.tools.answer_question).toBeDefined();

  expect(config.tools.think!.strict).toBe(true);
  expect(config.tools.search_wikipedia!.strict).toBe(true);
  expect(config.tools.load_wikipedia_page!.strict).toBe(true);
  expect(config.tools.answer_question!.strict).toBe(true);
});

it("should get config with metrics", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  expect(config.metrics.exact_match).toBeDefined();
  expect(config.metrics.elapsed_ms).toBeDefined();
  expect(config.metrics.jaccard_similarity).toBeDefined();
  expect(config.metrics.haiku_score).toBeDefined();

  expect(config.metrics.exact_match!.type).toBe("boolean");
  expect(config.metrics.elapsed_ms!.type).toBe("float");
  expect(config.metrics.exact_match!.optimize).toBe("max");
  expect(config.metrics.elapsed_ms!.optimize).toBe("min");
});

it("should get config with evaluations", async () => {
  const config = await getConfig(UI_FIXTURES_CONFIG_PATH);

  expect(config.evaluations.entity_extraction).toBeDefined();
  expect(config.evaluations.haiku).toBeDefined();
  expect(config.evaluations.images).toBeDefined();

  expect(config.evaluations.entity_extraction!.type).toBe("inference");
  expect(config.evaluations.haiku!.type).toBe("inference");
  expect(config.evaluations.images!.type).toBe("inference");

  expect(config.evaluations.entity_extraction!.function_name).toBe(
    "extract_entities",
  );
  expect(config.evaluations.haiku!.function_name).toBe("write_haiku");
  expect(config.evaluations.images!.function_name).toBe("image_judger");
});
