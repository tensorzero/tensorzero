import { describe, expect, test } from "vitest";
import {
  countDatapointsForEvaluation,
  countTotalEvaluationRuns,
  getEvaluationResults,
  getEvaluationRunInfo,
  getEvaluationRunInfos,
  getEvaluationRunInfosForDatapoint,
  getEvaluationsForDatapoint,
  getEvaluationStatistics,
  searchEvaluationRuns,
} from "./evaluations.server";
import type { ChatEvaluationResultWithVariant } from "./evaluations";
import { fail } from "assert";

describe("getEvaluationRunInfos", () => {
  test("should return correct run infos for specific evaluation run ids", async () => {
    const runInfos = await getEvaluationRunInfos(
      [
        "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        evaluation_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        most_recent_inference_date: "2025-03-23T21:56:17Z",
        variant_name: "llama_8b_initial_prompt",
      },
      {
        evaluation_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
    ]);
  });

  test("should return empty array when no matching run ids are found", async () => {
    const runInfos = await getEvaluationRunInfos(
      ["non-existent-id"],
      "extract_entities",
    );
    expect(runInfos).toEqual([]);
  });

  test("should handle a single run id correctly", async () => {
    const runInfos = await getEvaluationRunInfos(
      ["0195aef7-ec99-7312-924f-32b71c3496ee"],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        evaluation_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
    ]);
  });
});

describe("searchEvaluationRuns", () => {
  test("should return matching run ids when searching by evaluation_run_id prefix", async () => {
    const runIds = await searchEvaluationRuns(
      "entity_extraction",
      "extract_entities",
      "0195c5",
    );
    expect(runIds).toMatchObject([
      {
        evaluation_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        variant_name: "llama_8b_initial_prompt",
      },
    ]);
  });

  test("should return matching run ids when searching by variant_name for models", async () => {
    const runIds = await searchEvaluationRuns(
      "entity_extraction",
      "extract_entities",
      "gpt4o",
    );
    expect(runIds).toMatchObject([
      {
        evaluation_run_id: "0195f845-8f85-7822-b904-10630698f99c",
      },
      {
        evaluation_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        evaluation_run_id: "0195aef7-ec99-7312-924f-32b71c3496ee",
        variant_name: "gpt4o_initial_prompt",
      },
    ]);
  });

  test("should return matching run ids when searching by partial variant_name", async () => {
    const runIds = await searchEvaluationRuns(
      "haiku",
      "write_haiku",
      "initial",
    );
    expect(runIds).toMatchObject([
      {
        evaluation_run_id: "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        variant_name: "initial_prompt_gpt4o_mini",
      },
      {
        evaluation_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
        variant_name: "initial_prompt_haiku_3_5",
      },
    ]);
  });

  test("should handle case-insensitive search", async () => {
    const runIds = await searchEvaluationRuns(
      "entity_extraction",
      "extract_entities",
      "llama",
    );
    expect(runIds).toMatchObject([
      {
        evaluation_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        variant_name: "llama_8b_initial_prompt",
      },
    ]);
  });

  test("should return empty array when no matches found", async () => {
    const runIds = await searchEvaluationRuns(
      "entity_extraction",
      "extract_entities",
      "nonexistent",
    );
    expect(runIds).toEqual([]);
  });
});

describe("getEvaluationResults", () => {
  test("should return correct results for haiku evaluation", async () => {
    const results = await getEvaluationResults(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
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
      expect(result).toHaveProperty("evaluation_run_id");
      expect(result).toHaveProperty("input");
      expect(result).toHaveProperty("generated_output");
      expect(result).toHaveProperty("reference_output");
      expect(result).toHaveProperty("metric_name");
      expect(result).toHaveProperty("metric_value");
    });

    // Verify the evaluation_run_id is consistent across all results
    expect(
      results.every(
        (r) => r.evaluation_run_id === "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
      ),
    ).toBe(true);

    // Verify we have both metric types in the results
    const metricNames = new Set(results.map((r) => r.metric_name));
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ),
    ).toBe(true);
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
      ),
    ).toBe(true);
    // Verify that the number of distinct datapoint ids is 5
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(5);
  });

  test("should return correct results for entity_extraction evaluation that skips a staled datapoint", async () => {
    // There is a datapoint that was inserted and deleted before the last evaluation run after the first two.
    // We test here that it is not included and the data is ragged due to the datapoint at the top of the
    // table only having one evaluation run.
    const results = await getEvaluationResults(
      "extract_entities",
      "json",
      [
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
      ],
      [
        "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
        "0195aef7-ec99-7312-924f-32b71c3496ee",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
      2,
      0,
    );
    expect(results.length).toBe(8); // 1 datapoints * 3 evaluation runs * 2 metrics + 1 ragged datapoint * 1 evaluation run * 2 metrics
    // Verify that we have both metrics in the results
    const metricNames = new Set(results.map((r) => r.metric_name));
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
      ),
    ).toBe(true);
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
      ),
    ).toBe(true);
    // Verify that the number of distinct datapoint ids is 2
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(2);
  });

  test("should return correct results for ragged haiku evaluation", async () => {
    const results = await getEvaluationResults(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ],
      [
        "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
        "0195c498-1cbe-7ac0-b5b2-5856741f5890",
      ],
      5,
      0,
    );
    // Verify we get the expected number of results (18)
    // 18 is because the most recent datapoint is skipped because it is after all the evaluation runs
    // and the second most recent datapoint is only used in one of the evaluation runs
    // so it's 5 datapoints * 2 metrics * 2 evaluation runs - 1 skipped datapoint for 1 evaluation run * 2 metrics
    expect(results.length).toBe(18);

    // Check that each result has the expected structure
    results.forEach((result) => {
      expect(result).toHaveProperty("datapoint_id");
      expect(result).toHaveProperty("evaluation_run_id");
      expect(result).toHaveProperty("input");
      expect(result).toHaveProperty("generated_output");
      expect(result).toHaveProperty("reference_output");
      expect(result).toHaveProperty("metric_name");
      expect(result).toHaveProperty("metric_value");
    });

    // Verify the evaluation_run_id is consistent across all results
    expect(
      results.every(
        (r) =>
          r.evaluation_run_id === "0195aef7-96fe-7d60-a2e6-5a6ea990c425" ||
          r.evaluation_run_id === "0195c498-1cbe-7ac0-b5b2-5856741f5890",
      ),
    ).toBe(true);

    // Verify we have both metric types in the results
    const metricNames = new Set(results.map((r) => r.metric_name));
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ),
    ).toBe(true);
    expect(
      metricNames.has(
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
      ),
    ).toBe(true);
    // Verify that the number of distinct datapoint ids is 5
    const datapointIds = new Set(results.map((r) => r.datapoint_id));
    expect(datapointIds.size).toBe(5);
  });
});

describe("getEvaluationStatistics", () => {
  test("should return correct statistics for haiku evaluation", async () => {
    const statistics = await getEvaluationStatistics(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ],
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(statistics.length).toBe(2);
    expect(statistics[0].evaluation_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(statistics[0].metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
    );
    expect(statistics[0].datapoint_count).toBe(75);
    expect(statistics[0].mean_metric).toBe(0);
    expect(statistics[0].stderr_metric).toBe(0);
    expect(statistics[1].evaluation_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(statistics[1].datapoint_count).toBe(75);
    expect(statistics[1].mean_metric).toBeCloseTo(0.066667);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.02428);
  });

  test("should return correct statistics for entity_extraction evaluation", async () => {
    const statistics = await getEvaluationStatistics(
      "extract_entities",
      "json",
      [
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
      ],
      [
        "0195aef7-ec99-7312-924f-32b71c3496ee",
        "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      ],
    );
    expect(statistics.length).toBe(4); // 2 evaluation runs * 2 metrics
    expect(statistics[0].evaluation_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[0].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[0].datapoint_count).toBe(41);
    expect(statistics[0].mean_metric).toBeCloseTo(0.22);
    expect(statistics[0].stderr_metric).toBeCloseTo(0.0654);

    expect(statistics[1].evaluation_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[1].datapoint_count).toBe(41);
    expect(statistics[1].mean_metric).toBeCloseTo(0.8048);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.0627);

    expect(statistics[2].evaluation_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(statistics[2].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[2].datapoint_count).toBe(41);
    expect(statistics[2].mean_metric).toBeCloseTo(0.7805);
    expect(statistics[2].stderr_metric).toBeCloseTo(0.065);

    expect(statistics[3].evaluation_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(statistics[3].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[3].datapoint_count).toBe(41);
    expect(statistics[3].mean_metric).toBeCloseTo(0.54);
    expect(statistics[3].stderr_metric).toBeCloseTo(0.0788);
  });
});

describe("countDatapointsForEvaluation", () => {
  test("should return correct number of datapoints for haiku evaluation", async () => {
    const datapoints = await countDatapointsForEvaluation(
      "write_haiku",
      "chat",
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    // This should not include data that is after the evaluation run
    expect(datapoints).toBe(75);
  });

  test("should return correct number of datapoints for entity_extraction evaluation", async () => {
    const datapoints = await countDatapointsForEvaluation(
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

describe("countTotalEvaluationRuns", () => {
  test("should return correct number of evaluation runs", async () => {
    const runs = await countTotalEvaluationRuns();
    expect(runs).toBe(7);
  });
});

describe("getEvaluationRunInfo", () => {
  test("should return correct evaluation run info", async () => {
    const runs = await getEvaluationRunInfo();

    // Check the total number of runs
    expect(runs.length).toBe(7);

    // Check structure and content of the first row
    expect(runs[0]).toMatchObject({
      evaluation_run_id: "0195f845-8f85-7822-b904-10630698f99c",
      evaluation_name: "entity_extraction",
      function_name: "extract_entities",
      variant_name: "gpt4o_mini_initial_prompt",
      last_inference_timestamp: "2025-04-02T20:51:05Z",
      dataset_name: "foo",
    });

    // Check structure and content of another row
    expect(runs[3]).toMatchObject({
      evaluation_run_id: "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
      evaluation_name: "entity_extraction",
      function_name: "extract_entities",
      variant_name: "gpt4o_mini_initial_prompt",
    });

    // Verify that all items have the expected properties
    runs.forEach((run) => {
      expect(run).toHaveProperty("evaluation_run_id");
      expect(run).toHaveProperty("evaluation_name");
      expect(run).toHaveProperty("function_name");
      expect(run).toHaveProperty("variant_name");
      expect(run).toHaveProperty("last_inference_timestamp");

      // Check data types
      expect(typeof run.evaluation_run_id).toBe("string");
      expect(typeof run.evaluation_name).toBe("string");
      expect(typeof run.function_name).toBe("string");
      expect(typeof run.variant_name).toBe("string");
      expect(typeof run.last_inference_timestamp).toBe("string");
    });

    // Verify that the runs are sorted by evaluation_run_id in descending order
    // This verifies the ORDER BY clause is working
    expect(runs[0].evaluation_run_id > runs[1].evaluation_run_id).toBe(true);

    // Check for specific evaluation_names in the dataset
    const evaluationNames = runs.map((run) => run.evaluation_name);
    expect(evaluationNames).toContain("entity_extraction");
    expect(evaluationNames).toContain("haiku");

    // Check for specific function_names in the dataset
    const functionNames = runs.map((run) => run.function_name);
    expect(functionNames).toContain("extract_entities");
    expect(functionNames).toContain("write_haiku");

    // Check the last run in the result
    expect(runs[6]).toMatchObject({
      evaluation_run_id: "0195aef6-4ed4-7710-ae62-abb10744f153",
      evaluation_name: "haiku",
      function_name: "write_haiku",
      variant_name: "initial_prompt_haiku_3_5",
    });
  });
});

describe("getEvaluationsForDatapoint", () => {
  test("should return empty array for nonexistent datapoint", async () => {
    const evaluations = await getEvaluationsForDatapoint(
      "haiku",
      "0195d806-e43d-7f7e-bb05-f6dd0d95846f", // Nonexistent datapoint
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(evaluations).toEqual([]);
  });

  test("should return correct array for chat datapoint", async () => {
    const evaluations = await getEvaluationsForDatapoint(
      "haiku",
      "01936551-ffc8-7372-8991-0a2929d3f5b0", // Real datapoint
      ["0195aef7-96fe-7d60-a2e6-5a6ea990c425"],
    );
    expect(evaluations.length).toBe(2);

    // Check first evaluation result
    const first_evaluation = evaluations[0] as ChatEvaluationResultWithVariant;
    expect(first_evaluation.datapoint_id).toBe(
      "01936551-ffc8-7372-8991-0a2929d3f5b0",
    );
    expect(first_evaluation.evaluation_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(first_evaluation.variant_name).toBe("initial_prompt_gpt4o_mini");
    expect(first_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
    );
    expect(first_evaluation.metric_value).toBe("false");
    expect(first_evaluation.generated_output).toHaveLength(1);
    const first_evaluation_output = first_evaluation.generated_output[0];
    if (first_evaluation_output.type === "text") {
      expect(first_evaluation_output.text).toContain("Pegboard, a canvas");
    } else {
      fail("First evaluation result is not a text");
    }

    // Check second evaluation result
    const second_evaluation = evaluations[1] as ChatEvaluationResultWithVariant;
    expect(second_evaluation.datapoint_id).toBe(
      "01936551-ffc8-7372-8991-0a2929d3f5b0",
    );
    expect(second_evaluation.evaluation_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(second_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(second_evaluation.metric_value).toBe("false");
    expect(second_evaluation.input.messages).toHaveLength(1);
    const second_evaluation_input = second_evaluation.input;
    if (second_evaluation_input.messages[0].content[0].type === "text") {
      expect(
        second_evaluation_input.messages[0].content[0].value,
      ).toStrictEqual({
        topic: "pegboard",
      });
    } else {
      fail("Second evaluation result is not a text");
    }
  });

  test("should return correct array for json datapoint", async () => {
    const evaluations = await getEvaluationsForDatapoint(
      "entity_extraction",
      "019373bc-e6e0-7e50-8822-af9bacfafe9a", // Real json datapoint
      ["0195aef7-ec99-7312-924f-32b71c3496ee"],
    );
    expect(evaluations.length).toBe(2);

    // Sort evaluations by metric name to ensure consistent order
    const sortedEvaluations = [...evaluations].sort((a, b) =>
      a.metric_name.localeCompare(b.metric_name),
    );

    // Check first evaluation result
    const first_evaluation = sortedEvaluations[0];
    expect(first_evaluation.datapoint_id).toBe(
      "019373bc-e6e0-7e50-8822-af9bacfafe9a",
    );
    expect(first_evaluation.evaluation_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(first_evaluation.variant_name).toBe("gpt4o_initial_prompt");
    expect(first_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(first_evaluation.metric_value).toBeDefined();

    // Check that we have JSON input/output fields
    expect(typeof first_evaluation.input).toBe("object");
    expect(typeof first_evaluation.reference_output).toBe("object");
    expect(typeof first_evaluation.generated_output).toBe("object");

    // Check second evaluation result
    const second_evaluation = sortedEvaluations[1];
    expect(second_evaluation.datapoint_id).toBe(
      "019373bc-e6e0-7e50-8822-af9bacfafe9a",
    );
    expect(second_evaluation.evaluation_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(second_evaluation.variant_name).toBe("gpt4o_initial_prompt");
    expect(second_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(second_evaluation.metric_value).toBeDefined();
  });
});

describe("getEvaluationRunInfosForDatapoint", () => {
  test("should return correct evaluation run info for ragged json datapoint", async () => {
    const evaluationRunInfos = await getEvaluationRunInfosForDatapoint(
      "0195c4ff-bc90-7bf1-8b99-3ebcf9e2a6f0",
      "extract_entities",
    );
    expect(evaluationRunInfos.length).toBe(2);

    // Check that the evaluation run ids are correct
    const expected1 = {
      evaluation_run_id: "0195f845-8f85-7822-b904-10630698f99c",
      most_recent_inference_date: "2025-04-02T20:51:03Z",
      variant_name: "gpt4o_mini_initial_prompt",
    };
    const expected2 = {
      evaluation_run_id: "0195c501-8e6b-76f2-aa2c-d7d379fe22a5",
      most_recent_inference_date: "2025-03-23T21:56:10Z",
      variant_name: "llama_8b_initial_prompt",
    };
    expect(evaluationRunInfos).toHaveLength(2); // Ensure exactly two items
    expect(evaluationRunInfos).toEqual(
      expect.arrayContaining([expected1, expected2]),
    );
  });

  test("should return correct evaluation run info for ragged haiku datapoint", async () => {
    const evaluationRunInfos = await getEvaluationRunInfosForDatapoint(
      "01963691-7489-74b3-837f-e386de14c5f9",
      "write_haiku",
    );

    const expected = {
      evaluation_run_id: "01963691-9d3c-7793-a8be-3937ebb849c1",
      variant_name: "better_prompt_haiku_3_5",
      most_recent_inference_date: "2025-03-23T20:01:25Z",
    };
    expect(evaluationRunInfos).toHaveLength(3); // Ensure exactly one item
    expect(evaluationRunInfos).toEqual(expect.arrayContaining([expected]));
  });
});
