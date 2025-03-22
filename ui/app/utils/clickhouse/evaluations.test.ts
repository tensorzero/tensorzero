import { describe, expect, test } from "vitest";
import {
  countDatapointsForEval,
  getEvalResults,
  getEvalRunIds,
  getEvalStatistics,
} from "./evaluations.server";

// TODO: add fixtures and tests that handle joins that are not as clean and contain missing data.

describe("getEvalRunIds", () => {
  test("should return correct run ids for entity_extraction eval", async () => {
    const runIds = await getEvalRunIds("entity_extraction");
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

  test("should return correct run ids for haiku eval", async () => {
    const runIds = await getEvalRunIds("haiku");
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
    // Verify we get the expected number of results (20 based on the error output)
    expect(results.length).toBe(20);

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

  test("should return correct results for entity_extraction eval", async () => {
    const results = await getEvalResults(
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
      6,
      0,
    );
    expect(results.length).toBe(48); // 6 datapoints * 2 eval runs * 2 metrics
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
    expect(statistics[0].datapoint_count).toBe(150);
    expect(statistics[0].mean_metric).toBe(0);
    expect(statistics[0].stderr_metric).toBe(0);
    expect(statistics[1].eval_run_id).toBe(
      "0195aef7-96fe-7d60-a2e6-5a6ea990c425",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::eval_name::haiku::evaluator_name::topic_starts_with_f",
    );
    expect(statistics[1].datapoint_count).toBe(150);
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
    expect(statistics[0].datapoint_count).toBe(82);
    expect(statistics[0].mean_metric).toBeCloseTo(0.7805);
    expect(statistics[0].stderr_metric).toBeCloseTo(0.046);
    expect(statistics[1].eval_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[1].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports",
    );
    expect(statistics[1].datapoint_count).toBe(82);
    expect(statistics[1].mean_metric).toBeCloseTo(0.8048);
    expect(statistics[1].stderr_metric).toBeCloseTo(0.046);

    expect(statistics[2].eval_run_id).toBe(
      "0195aef8-36bf-7c02-b8a2-40d78049a4a0",
    );
    expect(statistics[2].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[2].datapoint_count).toBe(82);
    expect(statistics[2].mean_metric).toBeCloseTo(0.22);
    expect(statistics[2].stderr_metric).toBeCloseTo(0.046);

    expect(statistics[3].eval_run_id).toBe(
      "0195aef7-ec99-7312-924f-32b71c3496ee",
    );
    expect(statistics[3].metric_name).toBe(
      "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match",
    );
    expect(statistics[3].datapoint_count).toBe(82);
    expect(statistics[3].mean_metric).toBeCloseTo(0.54);
    expect(statistics[3].stderr_metric).toBeCloseTo(0.0554);
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
