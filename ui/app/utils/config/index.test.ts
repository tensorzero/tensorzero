import { expect, test } from "vitest";
import { loadConfig } from "./index.server";

test("parse e2e config", async () => {
  const validatedConfig = await loadConfig(
    "./../tensorzero-internal/tests/e2e/tensorzero.toml",
  );
  expect(validatedConfig).toBeDefined();
  // Test something in the gateway config
  expect(validatedConfig.gateway?.bind_address).toBe("0.0.0.0:3000");

  // Test something in the model config
  const azureProvider =
    validatedConfig.models["gpt-4o-mini-azure"].providers["azure"];
  if (azureProvider.type === "azure") {
    expect(azureProvider.deployment_id).toBe("gpt4o-mini-20240718");
  } else {
    throw new Error("Azure provider not found");
  }
  // Test a chat function
  const basicTestFunction = validatedConfig.functions["basic_test"];
  const openAiVariant = basicTestFunction.variants.openai;
  if (openAiVariant.type === "chat_completion") {
    expect(openAiVariant.model).toBe("gpt-4o-mini-2024-07-18");
  } else {
    throw new Error("OpenAI variant not found");
  }

  // Test a json function
  const jsonTestFunction = validatedConfig.functions["json_success"];
  if (jsonTestFunction.type === "json") {
    expect(jsonTestFunction.output_schema).toEqual({
      content: {
        additionalProperties: false,
        properties: {
          answer: {
            type: "string",
          },
        },
        required: ["answer"],
        type: "object",
      },
      path: "../../fixtures/config/functions/basic_test/output_schema.json",
    });
  } else {
    throw new Error("JSON function not found");
  }
  const gcpVertexGeminiProVariant =
    jsonTestFunction.variants["gcp-vertex-gemini-pro"];
  if (gcpVertexGeminiProVariant.type === "chat_completion") {
    expect(gcpVertexGeminiProVariant.model).toBe(
      "gemini-2.5-pro-preview-06-05",
    );
  } else {
    throw new Error("GCP Vertex Gemini Pro variant not found");
  }

  // Test a chat function with best of n sampling
  const bestOfNTestFunction = validatedConfig.functions["best_of_n"];
  const bestOfNVariant = bestOfNTestFunction.variants["best_of_n_variant"];
  if (bestOfNVariant.type === "experimental_best_of_n_sampling") {
    expect(bestOfNVariant.candidates).toEqual(["variant0", "variant1"]);
    const evaluator = bestOfNVariant.evaluator;
    expect(evaluator.system_template).toBe(
      "../../fixtures/config/functions/basic_test/prompt/system_template.minijinja",
    );
  } else {
    throw new Error("Best of N variant not found");
  }

  // Test a chat function with dynamic in context learning
  const basicTestDiclVariant =
    validatedConfig.functions["basic_test"].variants["dicl"];
  if (
    basicTestDiclVariant.type === "experimental_dynamic_in_context_learning"
  ) {
    expect(basicTestDiclVariant.model).toBe("gpt-4o-mini-2024-07-18");
    expect(basicTestDiclVariant.embedding_model).toBe("text-embedding-3-small");
    expect(basicTestDiclVariant.k).toBe(3);
  } else {
    throw new Error("Dynamic in context learning variant not found");
  }

  // Test a chat function with tools
  const weatherHelperFunction = validatedConfig.functions["weather_helper"];
  if (weatherHelperFunction.type === "chat") {
    expect(weatherHelperFunction.tools).toEqual(["get_temperature"]);
    expect(weatherHelperFunction.tool_choice).toBe("auto");
  } else {
    throw new Error("Weather helper function not found");
  }

  // Test a metric config
  const taskSuccessMetric = validatedConfig.metrics["task_success"];
  if (taskSuccessMetric.type === "boolean") {
    expect(taskSuccessMetric.optimize).toBe("max");
    expect(taskSuccessMetric.level).toBe("inference");
  } else {
    throw new Error("Task success metric not found");
  }

  // Test demonstration metric is added
  const demonstrationMetric = validatedConfig.metrics["demonstration"];
  expect(demonstrationMetric).toBeDefined();
  expect(demonstrationMetric.type).toBe("demonstration");
  expect(
    (demonstrationMetric as { type: "demonstration"; level: string }).level,
  ).toBe("inference");

  // Test a tool config
  const getTemperatureTool = validatedConfig.tools["get_temperature"];
  expect(getTemperatureTool.description).toBe(
    "Get the current temperature in a given location",
  );
  expect(getTemperatureTool.parameters).toBe(
    "../../fixtures/config/tools/get_temperature.json",
  );

  // Test evaluation configs are properly generated
  // Test that evaluations from the config are present
  expect(validatedConfig.evaluations).toBeDefined();
  expect(Object.keys(validatedConfig.evaluations).length).toBeGreaterThan(0);

  // Get a sample evaluation and check its structure
  const sampleEvaluation = validatedConfig.evaluations["entity_test"]; // Using an evaluation from the test config
  if (sampleEvaluation) {
    expect(sampleEvaluation.function_name).toBeDefined();
    expect(sampleEvaluation.evaluators).toBeDefined();

    // Check if evaluators are properly loaded
    expect(Object.keys(sampleEvaluation.evaluators).length).toBeGreaterThan(0);

    // If there's an exact_match evaluator, check its properties
    const exactMatchevaluation = Object.entries(
      sampleEvaluation.evaluators,
    ).find(([, evaluation_config]) => evaluation_config.type === "exact_match");
    if (exactMatchevaluation) {
      const [name, config] = exactMatchevaluation;
      expect(name).toBeDefined();
      expect(config.type).toBe("exact_match");
      if ("cutoff" in config) {
        expect(typeof config.cutoff).toBe("number");
      }
    }

    // If there's an llm_judge evaluator, check its properties
    const llmJudgeEvaluation = Object.entries(sampleEvaluation.evaluators).find(
      ([, evaluation_config]) => evaluation_config.type === "llm_judge",
    );
    if (llmJudgeEvaluation) {
      const [name, config] = llmJudgeEvaluation;
      expect(name).toBeDefined();
      expect(config.type).toBe("llm_judge");

      // Check for the generated function
      const functionName = `tensorzero::llm_judge::entity_test::${name}`;
      expect(validatedConfig.functions[functionName]).toBeDefined();

      if (validatedConfig.functions[functionName]) {
        const generatedFunction = validatedConfig.functions[functionName];
        if (generatedFunction.type === "json") {
          expect(generatedFunction.variants).toBeDefined();
          expect(
            Object.keys(generatedFunction.variants).length,
          ).toBeGreaterThan(0);
          expect(generatedFunction.output_schema).toBeDefined();
          expect(generatedFunction.user_schema).toBeDefined();
        }
      }

      // Check for the generated metric
      const metricName = `tensorzero::evaluation_name::entity_test::evaluator_name::${name}`;
      expect(validatedConfig.metrics[metricName]).toBeDefined();

      if (validatedConfig.metrics[metricName]) {
        const generatedMetric = validatedConfig.metrics[metricName];
        if (
          generatedMetric.type === "float" ||
          generatedMetric.type === "boolean"
        ) {
          expect(generatedMetric.optimize).toMatch(/min|max/);
          expect(generatedMetric.level).toBe("inference");
        }
      }
    }
  }
});

test("parse empty config", async () => {
  const validatedConfig = await loadConfig("fixtures/config/empty.toml");
  expect(validatedConfig).toBeDefined();
});

test("parse fixture config with evaluations", async () => {
  const validatedConfig = await loadConfig("fixtures/config/tensorzero.toml");
  expect(validatedConfig).toBeDefined();
  expect(validatedConfig.evaluations).toBeDefined();

  // Check entity_extraction evaluation
  const entityExtractionEvaluation =
    validatedConfig.evaluations["entity_extraction"];
  expect(entityExtractionEvaluation).toBeDefined();
  expect(entityExtractionEvaluation.function_name).toBe("extract_entities");

  // Check exact_match evaluator
  const exactMatchEvaluator =
    entityExtractionEvaluation.evaluators["exact_match"];
  expect(exactMatchEvaluator).toBeDefined();
  expect(exactMatchEvaluator.type).toBe("exact_match");
  expect(exactMatchEvaluator.cutoff).toBe(0.6);

  // Check count_sports evaluator (llm_judge)
  const countSportsEvaluator =
    entityExtractionEvaluation.evaluators["count_sports"];
  expect(countSportsEvaluator).toBeDefined();
  if (countSportsEvaluator.type === "llm_judge") {
    expect(countSportsEvaluator.output_type).toBe("float");
    expect(countSportsEvaluator.optimize).toBe("min");
    expect(countSportsEvaluator.cutoff).toBe(0.5);
  }

  // Check for generated function for count_sports
  const countSportsFunctionName =
    "tensorzero::llm_judge::entity_extraction::count_sports";
  expect(validatedConfig.functions[countSportsFunctionName]).toBeDefined();

  // Check function properties
  const countSportsFunction =
    validatedConfig.functions[countSportsFunctionName];
  if (countSportsFunction && countSportsFunction.type === "json") {
    expect(countSportsFunction.variants).toBeDefined();

    // Check variants
    const miniVariant = countSportsFunction.variants["mini"];
    expect(miniVariant).toBeDefined();
    if (miniVariant && miniVariant.type === "chat_completion") {
      expect(miniVariant.model).toBe("openai::gpt-4o-mini-2024-07-18");
      expect(miniVariant.weight).toBe(1.0); // Should be 1.0 since active=true
    }
  }

  // Check for generated metric
  const countSportsMetricName =
    "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports";
  expect(validatedConfig.metrics[countSportsMetricName]).toBeDefined();

  // Check metric properties
  const countSportsMetric = validatedConfig.metrics[countSportsMetricName];
  if (countSportsMetric && countSportsMetric.type === "float") {
    expect(countSportsMetric.optimize).toBe("min");
    expect(countSportsMetric.level).toBe("inference");
  }

  // Check haiku evaluation
  const haikuEvaluation = validatedConfig.evaluations["haiku"];
  expect(haikuEvaluation).toBeDefined();
  expect(haikuEvaluation.function_name).toBe("write_haiku");

  // Check topic_starts_with_f evaluator
  const topicStartsWithFEvaluator =
    haikuEvaluation.evaluators["topic_starts_with_f"];
  expect(topicStartsWithFEvaluator).toBeDefined();
  if (
    topicStartsWithFEvaluator &&
    topicStartsWithFEvaluator.type === "llm_judge"
  ) {
    expect(topicStartsWithFEvaluator.output_type).toBe("boolean");
  }

  // Check for generated function for topic_starts_with_f
  const topicStartsWithFFunctionName =
    "tensorzero::llm_judge::haiku::topic_starts_with_f";
  expect(validatedConfig.functions[topicStartsWithFFunctionName]).toBeDefined();

  // Check for generated metric
  const topicStartsWithFMetricName =
    "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f";
  expect(validatedConfig.metrics[topicStartsWithFMetricName]).toBeDefined();

  // Check metric properties
  const topicStartsWithFMetric =
    validatedConfig.metrics[topicStartsWithFMetricName];
  if (topicStartsWithFMetric && topicStartsWithFMetric.type === "boolean") {
    expect(topicStartsWithFMetric.optimize).toBe("min");
  }
});
