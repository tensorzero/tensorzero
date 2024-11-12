import { readFileSync } from "fs";
import { parse } from "smol-toml";
import { expect, test } from "vitest";
import { Config } from "./index";

test("parse e2e config", () => {
  const tomlContent = readFileSync(
    "./../gateway/tests/e2e/tensorzero.toml",
    "utf-8",
  );
  const parsedConfig = parse(tomlContent);
  const validatedConfig = Config.parse(parsedConfig);
  expect(validatedConfig).toBeDefined();
  // Test something in the gateway config
  expect(validatedConfig.gateway.bind_address).toBe("0.0.0.0:3000");

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
    expect(jsonTestFunction.output_schema).toBe(
      "../../fixtures/config/functions/basic_test/output_schema.json",
    );
  } else {
    throw new Error("JSON function not found");
  }
  const gcpVertexGeminiProVariant =
    jsonTestFunction.variants["gcp-vertex-gemini-pro"];
  if (gcpVertexGeminiProVariant.type === "chat_completion") {
    expect(gcpVertexGeminiProVariant.model).toBe("gemini-1.5-pro-001");
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

  // Test a tool config
  const getTemperatureTool = validatedConfig.tools["get_temperature"];
  expect(getTemperatureTool.description).toBe(
    "Get the current temperature in a given location",
  );
  expect(getTemperatureTool.parameters).toBe(
    "../../fixtures/config/tools/get_temperature.json",
  );
});
