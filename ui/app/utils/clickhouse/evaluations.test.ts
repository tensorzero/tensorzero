import { describe, expect, test } from "vitest";
import {
  countDatapointsForEvaluation,
  countTotalEvaluationRuns,
  getEvaluationRunInfo,
  getEvaluationRunInfos,
  getEvaluationRunInfosForDatapoint,
  getEvaluationsForDatapoint,
  getEvaluationStatistics,
  getEvaluationResults,
  searchEvaluationRuns,
} from "./evaluations.server";
import type { ChatEvaluationResultWithVariant } from "./evaluations";
import { fail } from "assert";

describe("getEvaluationRunInfos", () => {
  test("should return correct run infos for specific evaluation run ids", async () => {
    const evaluation_run_id1 = "0196368f-19bd-7082-a677-1c0bf346ff24";
    const evaluation_run_id2 = "0196368e-53a8-7e82-a88d-db7086926d81";

    const runInfos = await getEvaluationRunInfos(
      [evaluation_run_id1, evaluation_run_id2],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        evaluation_run_id: evaluation_run_id1,
        most_recent_inference_date: "2025-04-14T23:07:50Z",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        evaluation_run_id: evaluation_run_id2,
        most_recent_inference_date: "2025-04-14T23:06:59Z",
        variant_name: "gpt4o_initial_prompt",
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
    const evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";
    const runInfos = await getEvaluationRunInfos(
      [evaluation_run_id],
      "extract_entities",
    );
    expect(runInfos).toMatchObject([
      {
        evaluation_run_id,
        variant_name: "gpt4o_mini_initial_prompt",
      },
    ]);
  });
});

describe("searchEvaluationRuns", () => {
  test("should return matching run ids when searching by evaluation_run_id prefix", async () => {
    const runIds = await searchEvaluationRuns(
      "entity_extraction",
      "extract_entities",
      "46ff24",
    );
    expect(runIds).toMatchObject([
      {
        evaluation_run_id: "0196368f-19bd-7082-a677-1c0bf346ff24",
        variant_name: "gpt4o_mini_initial_prompt",
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
        evaluation_run_id: "0196374c-2b06-7f50-b187-80c15cec5a1f",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        evaluation_run_id: "0196368f-19bd-7082-a677-1c0bf346ff24",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        evaluation_run_id: "0196368e-53a8-7e82-a88d-db7086926d81",
        variant_name: "gpt4o_initial_prompt",
      },
      {
        evaluation_run_id: "0196367b-c0bb-7f90-b651-f90eb9fba8f3",
        variant_name: "gpt4o_mini_initial_prompt",
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
        evaluation_run_id: "01963690-dff2-7cd3-b724-62fb705772a1",
        variant_name: "initial_prompt_gpt4o_mini",
      },
      {
        evaluation_run_id: "0196367a-702c-75f3-b676-d6ffcc7370a1",
        variant_name: "initial_prompt_gpt4o_mini",
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
        evaluation_run_id: "0196367b-1739-7483-b3f4-f3b0a4bda063",
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
    const evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";
    const results = await getEvaluationResults(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ],
      [evaluation_run_id],
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
      expect(result).toHaveProperty("name");
      expect(result).toHaveProperty("generated_output");
      expect(result).toHaveProperty("reference_output");
      expect(result).toHaveProperty("metric_name");
      expect(result).toHaveProperty("metric_value");
    });

    // Verify the evaluation_run_id is consistent across all results
    expect(
      results.every((r) => r.evaluation_run_id === evaluation_run_id),
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
    const evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";
    const results = await getEvaluationResults(
      "extract_entities",
      "json",
      [
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
      ],
      [evaluation_run_id],
      2,
      0,
    );
    expect(results.length).toBe(4); // 1 datapoints * 1 evaluation runs * 2 metrics + 1 ragged datapoint * 1 evaluation run * 2 metrics
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
    const evaluation_run_id1 = "0196374b-04a3-7013-9049-e59ed5fe3f74";
    const evaluation_run_id2 = "01963691-9d3c-7793-a8be-3937ebb849c1";
    const results = await getEvaluationResults(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ],
      [evaluation_run_id1, evaluation_run_id2],
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
      expect(result).toHaveProperty("name");
      expect(result).toHaveProperty("generated_output");
      expect(result).toHaveProperty("reference_output");
      expect(result).toHaveProperty("metric_name");
      expect(result).toHaveProperty("metric_value");
    });

    // Verify the evaluation_run_id is consistent across all results
    expect(
      results.every(
        (r) =>
          r.evaluation_run_id === evaluation_run_id1 ||
          r.evaluation_run_id === evaluation_run_id2,
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
    const evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";
    const statistics = await getEvaluationStatistics(
      "write_haiku",
      "chat",
      [
        "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
        "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
      ],
      [evaluation_run_id],
    );
    expect(statistics.length).toBe(2);
    expect(statistics[0].evaluation_run_id).toBe(evaluation_run_id);
    expect(statistics[0].metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
    );
    expect(statistics[0].datapoint_count).toBe(77);
    expect(statistics[0].mean_metric).toBeCloseTo(0);
    expect(statistics[0].stderr_metric).toBeCloseTo(0);
    expect(statistics[1].evaluation_run_id).toBe(evaluation_run_id);
    expect(statistics[1].metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(statistics[1].datapoint_count).toBe(77);
    expect(statistics[1].mean_metric).toBeCloseTo(0.064);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.028);
  });

  test("should return correct statistics for entity_extraction evaluation", async () => {
    const evaluation_run_id1 = "0196368f-19bd-7082-a677-1c0bf346ff24";
    const evaluation_run_id2 = "0196368e-53a8-7e82-a88d-db7086926d81";
    const statistics = await getEvaluationStatistics(
      "extract_entities",
      "json",
      [
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
      ],
      [evaluation_run_id1, evaluation_run_id2],
    );
    expect(statistics.length).toBe(4); // 2 evaluation runs * 2 metrics
    expect(statistics[0].evaluation_run_id).toBe(evaluation_run_id1);
    expect(statistics[0].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[0].datapoint_count).toBe(41);
    expect(statistics[0].mean_metric).toBeCloseTo(0.78);
    expect(statistics[0].stderr_metric).toBeCloseTo(0.07);

    expect(statistics[1].evaluation_run_id).toBe(evaluation_run_id1);
    expect(statistics[1].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[1].datapoint_count).toBe(41);
    expect(statistics[1].mean_metric).toBeCloseTo(0.1);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.05);

    expect(statistics[2].evaluation_run_id).toBe(evaluation_run_id2);
    expect(statistics[2].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[2].datapoint_count).toBe(42);
    expect(statistics[2].mean_metric).toBeCloseTo(0.762);
    expect(statistics[2].stderr_metric).toBeCloseTo(0.0665);

    expect(statistics[3].evaluation_run_id).toBe(evaluation_run_id2);
    expect(statistics[3].metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[3].datapoint_count).toBe(42);
    expect(statistics[3].mean_metric).toBeCloseTo(0.524);
    expect(statistics[3].stderr_metric).toBeCloseTo(0.08);
  });
});

describe("countDatapointsForEvaluation", () => {
  test("should return correct number of datapoints for haiku evaluation", async () => {
    const datapoints = await countDatapointsForEvaluation(
      "write_haiku",
      "chat",
      ["01963690-dff2-7cd3-b724-62fb705772a1"],
    );
    // This should not include data that is after the evaluation run
    expect(datapoints).toBe(77);
  });

  test("should return correct number of datapoints for entity_extraction evaluation", async () => {
    const datapoints = await countDatapointsForEvaluation(
      "extract_entities",
      "json",
      ["0196368f-19bd-7082-a677-1c0bf346ff24"],
    );
    expect(datapoints).toBe(41);
  });
});

describe("countTotalEvaluationRuns", () => {
  test("should return correct number of evaluation runs", async () => {
    const runs = await countTotalEvaluationRuns();
    expect(runs).toBe(9);
  });
});

describe("getEvaluationRunInfo", () => {
  test("should return correct evaluation run info", async () => {
    const runs = await getEvaluationRunInfo();

    // Check the total number of runs
    expect(runs.length).toBe(9);

    // Check structure and content of the first row
    expect(runs[0]).toMatchObject({
      dataset_name: "foo",
      evaluation_name: "entity_extraction",
      evaluation_run_id: "0196374c-2b06-7f50-b187-80c15cec5a1f",
      function_name: "extract_entities",
      last_inference_timestamp: "2025-04-15T02:34:21Z",
      variant_name: "gpt4o_mini_initial_prompt",
    });
    // Check structure and content of another row
    expect(runs[3]).toMatchObject({
      evaluation_name: "haiku",
      evaluation_run_id: "01963690-dff2-7cd3-b724-62fb705772a1",
      function_name: "write_haiku",
      variant_name: "initial_prompt_gpt4o_mini",
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
      evaluation_name: "entity_extraction",
      evaluation_run_id: "0196367b-c0bb-7f90-b651-f90eb9fba8f3",
      function_name: "extract_entities",
      variant_name: "gpt4o_mini_initial_prompt",
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
    const datapoint_id = "0196374a-d03f-7420-9da5-1561cba71ddb";
    const evaluation_run_id = "0196374b-04a3-7013-9049-e59ed5fe3f74";
    const evaluations = await getEvaluationsForDatapoint(
      "haiku",
      datapoint_id,
      [evaluation_run_id],
    );
    expect(evaluations.length).toBe(2);

    // Check first evaluation result
    const first_evaluation = evaluations[0] as ChatEvaluationResultWithVariant;
    expect(first_evaluation.datapoint_id).toBe(datapoint_id);
    expect(first_evaluation.evaluation_run_id).toBe(evaluation_run_id);
    expect(first_evaluation.variant_name).toBe("better_prompt_haiku_3_5");
    expect(first_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(first_evaluation.metric_value).toBe("false");
    expect(first_evaluation.generated_output).toHaveLength(1);
    const first_evaluation_output = first_evaluation.generated_output[0];
    if (first_evaluation_output.type === "text") {
      expect(first_evaluation_output.text).toContain("Swallowing moonlight");
    } else {
      fail("First evaluation result is not a text");
    }

    // Check second evaluation result
    const second_evaluation = evaluations[1] as ChatEvaluationResultWithVariant;
    expect(second_evaluation.datapoint_id).toBe(datapoint_id);
    expect(second_evaluation.evaluation_run_id).toBe(evaluation_run_id);
    expect(second_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::haiku::evaluator_name::exact_match",
    );
    expect(second_evaluation.metric_value).toBe("true");
    expect(second_evaluation.input.messages).toHaveLength(1);
    const second_evaluation_input = second_evaluation.input;
    if (second_evaluation_input.messages[0].content[0].type === "template") {
      expect(
        second_evaluation_input.messages[0].content[0].arguments,
      ).toStrictEqual({
        topic: "sheet",
      });
    } else {
      fail("Second evaluation result is not a text");
    }
  });

  test("should return correct array for json datapoint", async () => {
    const datapoint_id = "0193994e-5560-7610-a3a0-45fdd59338aa";
    const evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";
    const evaluations = await getEvaluationsForDatapoint(
      "entity_extraction",
      datapoint_id,
      [evaluation_run_id],
    );
    expect(evaluations.length).toBe(2);

    // Sort evaluations by metric name to ensure consistent order
    const sortedEvaluations = [...evaluations].sort((a, b) =>
      a.metric_name.localeCompare(b.metric_name),
    );

    // Check first evaluation result
    const first_evaluation = sortedEvaluations[0];
    expect(first_evaluation.datapoint_id).toBe(datapoint_id);
    expect(first_evaluation.evaluation_run_id).toBe(evaluation_run_id);
    expect(first_evaluation.variant_name).toBe("gpt4o_mini_initial_prompt");
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
    expect(second_evaluation.datapoint_id).toBe(datapoint_id);
    expect(second_evaluation.evaluation_run_id).toBe(evaluation_run_id);
    expect(second_evaluation.variant_name).toBe("gpt4o_mini_initial_prompt");
    expect(second_evaluation.metric_name).toBe(
      "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(second_evaluation.metric_value).toBeDefined();
  });
});

describe("getEvaluationRunInfosForDatapoint", () => {
  test("should return correct evaluation run info for ragged json datapoint", async () => {
    const evaluationRunInfos = await getEvaluationRunInfosForDatapoint(
      "0196368e-0b64-7321-ab5b-c32eefbf3e9f",
      "extract_entities",
    );
    // Check that the evaluation run ids are correct
    const expected1 = {
      evaluation_run_id: "0196368e-53a8-7e82-a88d-db7086926d81",
      most_recent_inference_date: "2025-04-14T23:06:59Z",
      variant_name: "gpt4o_initial_prompt",
    };
    expect(evaluationRunInfos).toHaveLength(1); // Ensure exactly one item
    expect(evaluationRunInfos).toEqual([expected1]);
  });

  test("should return correct evaluation run info for ragged haiku datapoint", async () => {
    const evaluationRunInfos = await getEvaluationRunInfosForDatapoint(
      "0196374a-d03f-7420-9da5-1561cba71ddb",
      "write_haiku",
    );

    const expected = {
      evaluation_run_id: "0196374b-04a3-7013-9049-e59ed5fe3f74",
      variant_name: "better_prompt_haiku_3_5",
      most_recent_inference_date: "2025-04-15T02:33:05Z",
    };
    expect(evaluationRunInfos).toHaveLength(1); // Ensure exactly one item
    expect(evaluationRunInfos).toEqual(expect.arrayContaining([expected]));
  });
});
