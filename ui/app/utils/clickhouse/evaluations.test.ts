import { describe, expect, test } from "vitest";
import {
  getEvaluationsForDatapoint,
  getEvaluationResults,
} from "./evaluations.server";
import type { ChatEvaluationResultWithVariant } from "./evaluations";
import { fail } from "assert";

// These tests still provide value since they validate the parsed results; we will remove them once we start returning
// structured objects from the gateway (instead of strings that the UI parses).
describe("getEvaluationResults", () => {
  test("should return correct results for haiku evaluation", async () => {
    const evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";
    const results = await getEvaluationResults(
      "haiku",
      "write_haiku",
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
      "entity_extraction",
      "extract_entities",
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
      "haiku",
      "write_haiku",
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
      (a.metric_name ?? "").localeCompare(b.metric_name ?? ""),
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
