import { describe, it, expect } from "vitest";
import { TensorZeroClient } from "../index.js";

describe("TensorZeroClient Integration Tests", () => {
  it("should be able to import TensorZeroClient", () => {
    expect(TensorZeroClient).toBeDefined();
    expect(typeof TensorZeroClient).toBe("function");
  });

  it("client creation throws on a bad config path", async () => {
    // This should throw an error that contains "Failed to parse config: Internal TensorZero Error: Config file not found: "foo""
    await expect(
      async () => await TensorZeroClient.build("foo"),
    ).rejects.toThrow(
      'Failed to parse config: Internal TensorZero Error: Config file not found: "foo"',
    );
  });

  it("should have required methods and initialize without credentials", async () => {
    const client = await buildClient();
    expect(typeof client.experimentalLaunchOptimizationWorkflow).toBe(
      "function",
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
      extractEntitiesConfig.variants,
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
      // Assert on variant templates
      if (
        variant!.inner.type === "chat_completion" &&
        variant!.inner.system_template
      ) {
        expect(typeof variant!.inner.system_template).toBe("object");
        expect(variant!.inner.system_template.path).toBeDefined();
        expect(variant!.inner.system_template.contents).toBeDefined();
        expect(variant!.inner.system_template.path).toContain("template");
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
      generateSecretConfig.variants,
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

  it("should get float metric config with episode level", async () => {
    const client = await buildClient();
    const metricConfig = await client.getMetricConfig(
      "jaccard_similarity_episode",
    );
    expect(metricConfig).toBeDefined();
    expect(metricConfig.type).toBe("float");
    expect(metricConfig.level).toBe("episode");
    expect(metricConfig.optimize).toBe("max");
  });

  it("should get metric config with min optimization", async () => {
    const client = await buildClient();
    const metricConfig = client.getMetricConfig("elapsed_ms");
    expect(metricConfig).toBeDefined();
    expect(metricConfig.type).toBe("float");
    expect(metricConfig.level).toBe("episode");
    expect(metricConfig.optimize).toBe("min");
  });

  it("should throw error for non-existent metric", async () => {
    const client = await buildClient();
    expect(() => client.getMetricConfig("non_existent_metric")).toThrow();
  });

  it("should get chat function config with tools", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("multi_hop_rag_agent");
    expect(functionConfig).toBeDefined();
    expect(functionConfig.type).toBe("chat");
    if (functionConfig.type === "chat") {
      expect(functionConfig.tools).toEqual([
        "think",
        "search_wikipedia",
        "load_wikipedia_page",
        "answer_question",
      ]);
      expect(functionConfig.tool_choice).toBe("required");
      expect(functionConfig.parallel_tool_calls).toBe(true);
    }
  });

  it("should get function config with system schema", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("ask_question");
    expect(functionConfig).toBeDefined();
    expect(functionConfig.type).toBe("json");
    if (functionConfig.type === "json") {
      expect(functionConfig.system_schema).toBeDefined();
      expect(functionConfig.output_schema).toBeDefined();
    }
  });

  it("should get function config with user schema", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("write_haiku");
    expect(functionConfig).toBeDefined();
    expect(functionConfig.type).toBe("chat");
    if (functionConfig.type === "chat") {
      expect(functionConfig.user_schema).toBeDefined();
    }
  });

  it("should get function variant with temperature setting", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("generate_secret");
    expect(functionConfig).toBeDefined();
    const variantNames = Object.keys(functionConfig.variants);
    expect(variantNames.length).toBeGreaterThan(0);
    const variant = functionConfig.variants[variantNames[0]];
    expect(variant).toBeDefined();
    if (variant!.inner.type === "chat_completion") {
      expect(variant!.inner.temperature).toBe(1.5);
      expect(variant!.inner.system_template).toBeDefined();
      if (variant!.inner.system_template) {
        expect(typeof variant!.inner.system_template).toBe("object");
        expect(variant!.inner.system_template.path).toBeDefined();
        expect(variant!.inner.system_template.contents).toBeDefined();
        expect(variant!.inner.system_template.path).toContain("template");
      }
    }
  });

  it("should get function with dicl variant", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("extract_entities");
    expect(functionConfig).toBeDefined();
    const variants = functionConfig.variants;
    expect(variants.dicl).toBeDefined();
    const diclVariant = variants.dicl;
    expect(diclVariant!.inner.type).toBe("dicl");
    if (diclVariant!.inner.type === "dicl") {
      expect(diclVariant!.inner.k).toBe(10);
      expect(diclVariant!.inner.embedding_model).toBe("text-embedding-3-small");
    }
  });

  it("should get function with chain of thought variant", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("judge_answer");
    expect(functionConfig).toBeDefined();
    const variants = functionConfig.variants;
    const variantNames = Object.keys(variants);
    expect(variantNames.length).toBeGreaterThan(0);
    const variant = variants[variantNames[0]];
    expect(variant!.inner.type).toBe("chain_of_thought");

    // Assert on variant templates
    if (variant!.inner.type === "chain_of_thought") {
      expect(variant!.inner.system_template).toBeDefined();
      expect(variant!.inner.user_template).toBeDefined();
      if (variant!.inner.system_template) {
        expect(typeof variant!.inner.system_template).toBe("object");
        expect(variant!.inner.system_template.path).toBeDefined();
        expect(variant!.inner.system_template.contents).toBeDefined();
        expect(variant!.inner.system_template.path).toContain("template");
      }
      if (variant!.inner.user_template) {
        expect(typeof variant!.inner.user_template).toBe("object");
        expect(variant!.inner.user_template.path).toBeDefined();
        expect(variant!.inner.user_template.contents).toBeDefined();
        expect(variant!.inner.user_template.path).toContain("template");
      }
    }
  });

  it("should get function with different json_mode settings", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("extract_entities");
    expect(functionConfig).toBeDefined();
    const variants = functionConfig.variants;

    // Test strict json_mode
    const strictVariant = variants.gpt4o_mini_initial_prompt;
    expect(strictVariant).toBeDefined();
    if (strictVariant!.inner.type === "chat_completion") {
      expect(strictVariant!.inner.json_mode).toBe("strict");
    }
  });

  it("should throw error for non-existent function", async () => {
    const client = await buildClient();
    expect(() => client.getFunctionConfig("non_existent_function")).toThrow();
  });

  it("should get all function config fields for comprehensive coverage", async () => {
    const client = await buildClient();
    const functionConfig = client.getFunctionConfig("multi_hop_rag_agent");
    expect(functionConfig).toBeDefined();
    expect(functionConfig.type).toBe("chat");
    expect(functionConfig.variants).toBeDefined();

    // Check variant structure
    const variantNames = Object.keys(functionConfig.variants);
    expect(variantNames.length).toBeGreaterThan(0);

    for (const variantName of variantNames) {
      const variant = functionConfig.variants[variantName];
      expect(variant).toBeDefined();
      expect(variant!.inner).toBeDefined();
      expect(variant!.inner.type).toBeDefined();
    }
  });

  it("should be able to list functions", async () => {
    const client = await buildClient();
    const functions = client.listFunctions();
    expect(functions.length).toBe(16);
    expect(functions).toContain("extract_entities");
  });

  it("should be able to list metrics", async () => {
    const client = await buildClient();
    const metrics = client.listMetrics();
    expect(metrics.length).toBe(21);
    expect(metrics).toContain("exact_match");
  });

  it("should be able to list evaluations", async () => {
    const client = await buildClient();
    const evaluations = client.listEvaluations();
    expect(evaluations.length).toBe(3);
    expect(evaluations).toContain("entity_extraction");
    expect(evaluations).toContain("haiku");
    expect(evaluations).toContain("images");
  });

  it("should be able to get evaluation config", async () => {
    const client = await buildClient();
    const evaluationConfig = client.getEvaluationConfig("entity_extraction");
    expect(evaluationConfig).toBeDefined();
    expect(evaluationConfig.type).toBe("static");
    expect(evaluationConfig.function_name).toBe("extract_entities");
    expect(evaluationConfig.evaluators).toBeDefined();
    expect(evaluationConfig.evaluators.exact_match).toBeDefined();
    expect(evaluationConfig.evaluators.count_sports).toBeDefined();
  });

  it("should get evaluation config with exact match evaluator", async () => {
    const client = await buildClient();
    const evaluationConfig = client.getEvaluationConfig("entity_extraction");
    expect(evaluationConfig.evaluators.exact_match).toBeDefined();
    const exactMatchEvaluator = evaluationConfig.evaluators.exact_match!;
    expect(exactMatchEvaluator.type).toBe("exact_match");
    expect(exactMatchEvaluator.cutoff).toBe(0.6);
  });

  it("should get evaluation config with llm judge evaluator", async () => {
    const client = await buildClient();
    const evaluationConfig = client.getEvaluationConfig("entity_extraction");
    expect(evaluationConfig.evaluators.count_sports).toBeDefined();
    const llmJudgeEvaluator = evaluationConfig.evaluators.count_sports!;
    expect(llmJudgeEvaluator.type).toBe("llm_judge");
    if (llmJudgeEvaluator.type === "llm_judge") {
      expect(llmJudgeEvaluator.output_type).toBe("float");
      expect(llmJudgeEvaluator.optimize).toBe("min");
      expect(llmJudgeEvaluator.cutoff).toBe(0.5);
      expect(llmJudgeEvaluator.include.reference_output).toBe(false);
    }
  });

  it("should get haiku evaluation config", async () => {
    const client = await buildClient();
    const evaluationConfig = client.getEvaluationConfig("haiku");
    expect(evaluationConfig).toBeDefined();
    expect(evaluationConfig.type).toBe("static");
    expect(evaluationConfig.function_name).toBe("write_haiku");
    expect(evaluationConfig.evaluators.exact_match).toBeDefined();
    expect(evaluationConfig.evaluators.topic_starts_with_f).toBeDefined();

    const topicEvaluator = evaluationConfig.evaluators.topic_starts_with_f!;
    expect(topicEvaluator.type).toBe("llm_judge");
    if (topicEvaluator.type === "llm_judge") {
      expect(topicEvaluator.output_type).toBe("boolean");
      expect(topicEvaluator.optimize).toBe("min");
      expect(topicEvaluator.include.reference_output).toBe(false);
    }
  });

  it("should get images evaluation config with message format", async () => {
    const client = await buildClient();
    const evaluationConfig = client.getEvaluationConfig("images");
    expect(evaluationConfig).toBeDefined();
    expect(evaluationConfig.type).toBe("static");
    expect(evaluationConfig.function_name).toBe("image_judger");

    const honestAnswerEvaluator = evaluationConfig.evaluators.honest_answer!;
    expect(honestAnswerEvaluator.type).toBe("llm_judge");
    if (honestAnswerEvaluator.type === "llm_judge") {
      expect(honestAnswerEvaluator.input_format).toBe("messages");
      expect(honestAnswerEvaluator.output_type).toBe("boolean");
      expect(honestAnswerEvaluator.optimize).toBe("max");
      expect(honestAnswerEvaluator.cutoff).toBe(0.2);
    }

    const matchesReferenceEvaluator =
      evaluationConfig.evaluators.matches_reference!;
    expect(matchesReferenceEvaluator.type).toBe("llm_judge");
    if (matchesReferenceEvaluator.type === "llm_judge") {
      expect(matchesReferenceEvaluator.include.reference_output).toBe(true);
    }
  });

  it("should throw error for non-existent evaluation", async () => {
    const client = await buildClient();
    expect(() =>
      client.getEvaluationConfig("non_existent_evaluation"),
    ).toThrow();
  });

  it("should get full config structure", async () => {
    const client = await buildClient();
    const config = client.getConfig();
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
    const client = await buildClient();
    const config = client.getConfig();
    expect(config.gateway.debug).toBe(true);
  });

  it("should get config with models including shorthand", async () => {
    const client = await buildClient();
    const config = client.getConfig();

    // Check shorthand model exists
    expect(config.models["gpt-4o-mini-2024-07-18"]).toBeDefined();
    expect(config.models["llama-3.1-8b-instruct"]).toBeDefined();
    expect(
      config.models["ft:gpt-4o-mini-2024-07-18:tensorzero::ALHEaw1j"],
    ).toBeDefined();

    // Check routing arrays
    expect(config.models["gpt-4o-mini-2024-07-18"]!.routing).toEqual([
      "openai",
    ]);
    expect(config.models["llama-3.1-8b-instruct"]!.routing).toEqual([
      "fireworks",
    ]);
    expect(
      config.models["ft:gpt-4o-mini-2024-07-18:tensorzero::ALHEaw1j"]!.routing,
    ).toEqual(["openai"]);
  });

  it("should get config with embedding models", async () => {
    const client = await buildClient();
    const config = client.getConfig();

    expect(config.embedding_models["text-embedding-3-small"]).toBeDefined();
    expect(config.embedding_models["text-embedding-3-small"]!.routing).toEqual([
      "openai",
    ]);
  });

  it("should get config with comprehensive function coverage", async () => {
    const client = await buildClient();
    const config = client.getConfig();

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
    expect(
      Object.keys(config.functions.extract_entities!.variants).length,
    ).toBe(6);
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
    const client = await buildClient();
    const config = client.getConfig();

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
    const client = await buildClient();
    const config = client.getConfig();

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
    const client = await buildClient();
    const config = client.getConfig();

    expect(config.evaluations.entity_extraction).toBeDefined();
    expect(config.evaluations.haiku).toBeDefined();
    expect(config.evaluations.images).toBeDefined();

    expect(config.evaluations.entity_extraction!.type).toBe("static");
    expect(config.evaluations.haiku!.type).toBe("static");
    expect(config.evaluations.images!.type).toBe("static");

    expect(config.evaluations.entity_extraction!.function_name).toBe(
      "extract_entities",
    );
    expect(config.evaluations.haiku!.function_name).toBe("write_haiku");
    expect(config.evaluations.images!.function_name).toBe("image_judger");
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
  return await TensorZeroClient.build(
    "../../ui/fixtures/config/tensorzero.toml",
  );
}
