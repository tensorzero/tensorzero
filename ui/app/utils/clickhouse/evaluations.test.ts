import { describe, expect, test } from "vitest";
import { getEvalResults, getEvalRunIds } from "./evaluations.server";

describe("getEvalRunIds", () => {
  test("should return correct run ids for entity_extraction eval", async () => {
    const runIds = await getEvalRunIds("entity_extraction");
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
      {
        eval_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
    ]);
  });

  test("should return correct run ids for haiku eval", async () => {
    const runIds = await getEvalRunIds("haiku");
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
        variant_name: "initial_prompt_haiku_3_5",
      },
      {
        eval_run_id: "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        variant_name: "initial_prompt_gpt4o_mini",
      },
    ]);
  });
});

test("getEvalResults", async () => {
  const results = await getEvalResults(
    "foo",
    "write_haiku",
    "chat",
    [
      "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
      "tensorzero::eval_name::haiku::evaluator_name::exact_match",
    ],
    ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    5,
    0,
  );
  // Verify we get the expected number of results (10 based on the error output)
  expect(results.length).toBe(10);

  // Check that each result has the expected structure
  results.forEach((result) => {
    expect(result).toHaveProperty("datapoint_id");
    expect(result).toHaveProperty("eval_run_id");
    expect(result).toHaveProperty("input");
    expect(result).toHaveProperty("generated_output");
    expect(result).toHaveProperty("reference_output");
    expect(result).toHaveProperty("metric_name");
    expect(result).toHaveProperty("metric_value");
  });

  // Verify the eval_run_id is consistent across all results
  expect(
    results.every(
      (r) => r.eval_run_id === "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    ),
  ).toBe(true);

  // Verify we have both metric types in the results
  const metricNames = new Set(results.map((r) => r.metric_name));
  expect(
    metricNames.has(
      "tensorzero::eval_name::haiku::evaluator_name::exact_match",
    ),
  ).toBe(true);
  expect(
    metricNames.has(
      "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
    ),
  ).toBe(true);
});
