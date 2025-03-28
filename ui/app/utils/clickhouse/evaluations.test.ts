import { describe, expect, test } from "vitest";
import {
  countDatapointsForEval,
  countTotalEvalRuns,
  getEvalResults,
  getEvalRunInfo,
  getEvalRunInfos,
  getEvalsForDatapoint,
  getEvalStatistics,
  getMostRecentEvalInferenceDate,
  searchEvalRuns,
} from "./evaluations.server";
import type { ChatEvaluationResultWithVariant } from "./evaluations";
import { fail } from "assert";

describe("getEvalRunInfos", () => {
  test("should return correct run infos for specific eval run ids", async () => {
    const runInfos = await getEvalRunInfos(
      [
        "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        eval_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        variant_name: "llama_8b_initial_prompt",
      },
      {
        eval_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
    ]);
  });

  test("should return empty array when no matching run ids are found", async () => {
    const runInfos = await getEvalRunInfos(
      ["non-existent-id"],
      "extract_entities",
    );
    expect(runInfos).toEqual([]);
  });

  test("should handle a single run id correctly", async () => {
    const runInfos = await getEvalRunInfos(
      ["0195aef7-ec99-7312-924f-32b71c3496ee"],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        eval_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
    ]);
  });
});

describe("searchEvalRuns", () => {
  test("should return matching run ids when searching by eval_run_id prefix", async () => {
    const runIds = await searchEvalRuns(
      "entity_extraction",
      "extract_entities",
      "0195c5",
    );
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        variant_name: "llama_8b_initial_prompt",
      },
    ]);
  });

  test("should return matching run ids when searching by variant_name for models", async () => {
    const runIds = await searchEvalRuns(
      "entity_extraction",
      "extract_entities",
      "gpt4o",
    );
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        eval_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
    ]);
  });

  test("should return matching run ids when searching by partial variant_name", async () => {
    const runIds = await searchEvalRuns("haiku", "write_haiku", "initial");
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        variant_name: "initial_prompt_gpt4o_mini",
      },
      {
        eval_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
        variant_name: "initial_prompt_haiku_3_5",
      },
    ]);
  });

  test("should handle case-insensitive search", async () => {
    const runIds = await searchEvalRuns(
      "entity_extraction",
      "extract_entities",
      "llama",
    );
    expect(runIds).toMatchObject([
      {
        eval_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        variant_name: "llama_8b_initial_prompt",
      },
    ]);
  });

  test("should return empty array when no matches found", async () => {
    const runIds = await searchEvalRuns(
      "entity_extraction",
      "extract_entities",
      "nonexistent",
    );
    expect(runIds).toEqual([]);
  });
});

describe("getEvalResults", () => {
  test("should return correct results for haiku eval", async () => {
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
    // Verify we get the expected number of results (10 = 5 datapoints * 2 metrics)
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
    // Verify that the number of distinct datapoint ids is 5
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(5);
  });

  test("should return correct results for entity_extraction eval that skips a staled datapoint", async () => {
    // There is a datapoint that was inserted and deleted before the last eval run after the first two.
    // We test here that it is not included and the data is not ragged.
    const results = await getEvalResults(
      "foo",
      "extract_entities",
      "json",
      [
        "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
      ],
      [
        "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        "0195aef7-ec99-7312-924f-32b71c3496ee",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
      6,
      0,
    );
    expect(results.length).toBe(36); // 6 datapoints * 3 eval runs * 2 metrics
    // Verify that we have both metrics in the results
    const metricNames = new Set(results.map((r) => r.metric_name));
    expect(
      metricNames.has(
        "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
      ),
    ).toBe(true);
    expect(
      metricNames.has(
        "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
      ),
    ).toBe(true);
    // Verify that the number of distinct datapoint ids is 6
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(6);
  });

  test("should return correct results for ragged haiku eval", async () => {
    const results = await getEvalResults(
      "foo",
      "write_haiku",
      "chat",
      [
        "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::eval_name::haiku::evaluator_name::exact_match",
      ],
      [
        "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        "0195c498-1cbe-7ac0-b5b2-5856741f5890",
      ],
      5,
      0,
    );
    // Verify we get the expected number of results (18)
    // 18 is because the most recent datapoint is skipped because it is after all the eval runs
    // and the second most recent datapoint is only used in one of the eval runs
    // so it's 5 datapoints * 2 metrics * 2 eval runs - 1 skipped datapoint for 1 eval run * 2 metrics
    expect(results.length).toBe(18);

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
        (r) =>
          r.eval_run_id === "0195aef7-96fe-7d60-a2e6-5a6ea990c425" ||
          r.eval_run_id === "0195c498-1cbe-7ac0-b5b2-5856741f5890",
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
    // Verify that the number of distinct datapoint ids is 5
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(5);
  });
});

describe("getEvalStatistics", () => {
  test("should return correct statistics for haiku eval", async () => {
    const statistics = await getEvalStatistics(
      "foo",
      "write_haiku",
      "chat",
      [
        "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::eval_name::haiku::evaluator_name::exact_match",
      ],
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(statistics.length).toBe(2);
    expect(statistics[0].eval_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(statistics[0].metric_name).toBe(
      "tensorzero::eval_name::haiku::evaluator_name::exact_match",
    );
    expect(statistics[0].datapoint_count).toBe(75);
    expect(statistics[0].mean_metric).toBe(0);
    expect(statistics[0].stderr_metric).toBe(0);
    expect(statistics[1].eval_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(statistics[1].datapoint_count).toBe(75);
    expect(statistics[1].mean_metric).toBeCloseTo(0.066667);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.02428);
  });

  test("should return correct statistics for entity_extraction eval", async () => {
    const statistics = await getEvalStatistics(
      "foo",
      "extract_entities",
      "json",
      [
        "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
      ],
      [
        "0195aef7-ec99-7312-924f-32b71c3496ee",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
    );
    expect(statistics.length).toBe(4); // 2 eval runs * 2 metrics
    expect(statistics[0].eval_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(statistics[0].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[0].datapoint_count).toBe(41);
    expect(statistics[0].mean_metric).toBeCloseTo(0.7805);
    expect(statistics[0].stderr_metric).toBeCloseTo(0.065);
    expect(statistics[1].eval_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[1].datapoint_count).toBe(41);
    expect(statistics[1].mean_metric).toBeCloseTo(0.8048);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.0627);

    expect(statistics[2].eval_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[2].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[2].datapoint_count).toBe(41);
    expect(statistics[2].mean_metric).toBeCloseTo(0.22);
    expect(statistics[2].stderr_metric).toBeCloseTo(0.0654);

    expect(statistics[3].eval_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(statistics[3].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[3].datapoint_count).toBe(41);
    expect(statistics[3].mean_metric).toBeCloseTo(0.54);
    expect(statistics[3].stderr_metric).toBeCloseTo(0.0788);
  });
});

describe("countDatapointsForEval", () => {
  test("should return correct number of datapoints for haiku eval", async () => {
    const datapoints = await countDatapointsForEval(
      "foo",
      "write_haiku",
      "chat",
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    // This should not include data that is after the eval run
    expect(datapoints).toBe(75);
  });

  test("should return correct number of datapoints for entity_extraction eval", async () => {
    const datapoints = await countDatapointsForEval(
      "foo",
      "extract_entities",
      "json",
      [
        "0195aef7-ec99-7312-924f-32b71c3496ee",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
    );
    expect(datapoints).toBe(41);
  });
});

describe("countTotalEvalRuns", () => {
  test("should return correct number of eval runs", async () => {
    const runs = await countTotalEvalRuns();
    expect(runs).toBe(6);
  });
});

describe("getEvalRunInfo", () => {
  test("should return correct eval run info", async () => {
    const runs = await getEvalRunInfo();

    // Check the total number of runs
    expect(runs.length).toBe(6);

    // Check structure and content of the first row
    expect(runs[0]).toMatchObject({
      eval_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
      eval_name: "entity_extraction",
      function_name: "extract_entities",
      variant_name: "llama_8b_initial_prompt",
      last_inference_timestamp: "2025-03-23T21:56:17Z",
    });

    // Check structure and content of another row
    expect(runs[2]).toMatchObject({
      eval_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      eval_name: "entity_extraction",
      function_name: "extract_entities",
      variant_name: "gpt4o_mini_initial_prompt",
    });

    // Verify that all items have the expected properties
    runs.forEach((run) => {
      expect(run).toHaveProperty("eval_run_id");
      expect(run).toHaveProperty("eval_name");
      expect(run).toHaveProperty("function_name");
      expect(run).toHaveProperty("variant_name");
      expect(run).toHaveProperty("last_inference_timestamp");

      // Check data types
      expect(typeof run.eval_run_id).toBe("string");
      expect(typeof run.eval_name).toBe("string");
      expect(typeof run.function_name).toBe("string");
      expect(typeof run.variant_name).toBe("string");
      expect(typeof run.last_inference_timestamp).toBe("string");
    });

    // Verify that the runs are sorted by eval_run_id in descending order
    // This verifies the ORDER BY clause is working
    expect(runs[0].eval_run_id > runs[1].eval_run_id).toBe(true);

    // Check for specific eval_names in the dataset
    const evalNames = runs.map((run) => run.eval_name);
    expect(evalNames).toContain("entity_extraction");
    expect(evalNames).toContain("haiku");

    // Check for specific function_names in the dataset
    const functionNames = runs.map((run) => run.function_name);
    expect(functionNames).toContain("extract_entities");
    expect(functionNames).toContain("write_haiku");

    // Check the last run in the result
    expect(runs[5]).toMatchObject({
      eval_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
      eval_name: "haiku",
      function_name: "write_haiku",
      variant_name: "initial_prompt_haiku_3_5",
    });
  });
});

describe("getMostRecentEvalInferenceDate", () => {
  test("should return correct last inference timestamp", async () => {
    const timestamps = await getMostRecentEvalInferenceDate([
      "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
    ]);
    expect(timestamps).toEqual(
      new Map([
        [
          "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
          new Date("2025-03-23T21:56:27.000Z"),
        ],
      ]),
    );
  });

  test("should return run timestamp if no inference id is found", async () => {
    const timestamps = await getMostRecentEvalInferenceDate([
      "0195c501-8e6b-76f2-aa2c-ffffffffffff",
    ]);
    expect(timestamps).toEqual(
      new Map([
        [
          "0195c501-8e6b-76f2-aa2c-ffffffffffff",
          new Date("2025-03-23T21:56:08.427Z"),
        ],
      ]),
    );
  });

  test("handles multiple eval run ids", async () => {
    const timestamps = await getMostRecentEvalInferenceDate([
      "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    ]);
    expect(timestamps).toEqual(
      new Map([
        [
          "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
          new Date("2025-03-23T21:56:27.000Z"),
        ],
        [
          "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
          new Date("2025-03-19T15:14:19.000Z"),
        ],
      ]),
    );
  });
});

describe("getEvalsForDatapoint", () => {
  test("should return empty array for nonexistent datapoint", async () => {
    const evals = await getEvalsForDatapoint(
      "haiku",
      "0195d806-e43d-7f7e-bb05-f6dd0d95846f", // Nonexistent datapoint
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(evals).toEqual([]);
  });

  test("should return correct array for chat datapoint", async () => {
    const evals = await getEvalsForDatapoint(
      "haiku",
      "01936551-ffc8-7372-8991-0a2929d3f5b0", // Real datapoint
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(evals.length).toBe(2);

    // Check first evaluation result
    const first_eval = evals[0] as ChatEvaluationResultWithVariant;
    expect(first_eval.datapoint_id).toBe(
      "01936551-ffc8-7372-8991-0a2929d3f5b0",
    );
    expect(first_eval.eval_run_id).toBe("0195aef7-96fe-7d60-a2e6-5a6ea990c425");
    expect(first_eval.variant_name).toBe("initial_prompt_gpt4o_mini");
    expect(first_eval.metric_name).toBe(
      "tensorzero::eval_name::haiku::evaluator_name::exact_match",
    );
    expect(first_eval.metric_value).toBe("false");
    expect(first_eval.generated_output).toHaveLength(1);
    const first_eval_output = first_eval.generated_output[0];
    if (first_eval_output.type === "text") {
      expect(first_eval_output.text).toContain("Pegboard, a canvas");
    } else {
      fail("First evaluation result is not a text");
    }

    // Check second evaluation result
    const second_eval = evals[1] as ChatEvaluationResultWithVariant;
    expect(second_eval.datapoint_id).toBe(
      "01936551-ffc8-7372-8991-0a2929d3f5b0",
    );
    expect(second_eval.eval_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(second_eval.metric_name).toBe(
      "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(second_eval.metric_value).toBe("false");
    expect(second_eval.input.messages).toHaveLength(1);
    const second_eval_input = second_eval.input;
    if (second_eval_input.messages[0].content[0].type === "text") {
      expect(second_eval_input.messages[0].content[0].value).toStrictEqual({
        topic: "pegboard",
      });
    } else {
      fail("Second evaluation result is not a text");
    }
  });

  test("should return correct array for json datapoint", async () => {
    const evals = await getEvalsForDatapoint(
      "entity_extraction",
      "019373bc-e6e0-7e50-8822-af9bacfafe9a", // Real json datapoint
      ["0195aef7-ec99-7312-924f-32b71c3496ee"],
    );
    expect(evals.length).toBe(2);

    // Sort evaluations by metric name to ensure consistent order
    const sortedEvals = [...evals].sort((a, b) =>
      a.metric_name.localeCompare(b.metric_name),
    );

    // Check first evaluation result
    const first_eval = sortedEvals[0];
    expect(first_eval.datapoint_id).toBe(
      "019373bc-e6e0-7e50-8822-af9bacfafe9a",
    );
    expect(first_eval.eval_run_id).toBe("0195aef7-ec99-7312-924f-32b71c3496ee");
    expect(first_eval.variant_name).toBe("gpt4o_initial_prompt");
    expect(first_eval.metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(first_eval.metric_value).toBeDefined();

    // Check that we have JSON input/output fields
    expect(typeof first_eval.input).toBe("object");
    expect(typeof first_eval.reference_output).toBe("object");
    expect(typeof first_eval.generated_output).toBe("object");

    // Check second evaluation result
    const second_eval = sortedEvals[1];
    expect(second_eval.datapoint_id).toBe(
      "019373bc-e6e0-7e50-8822-af9bacfafe9a",
    );
    expect(second_eval.eval_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(second_eval.variant_name).toBe("gpt4o_initial_prompt");
    expect(second_eval.metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(second_eval.metric_value).toBeDefined();
  });
});
