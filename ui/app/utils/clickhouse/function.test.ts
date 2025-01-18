import { expect, test, describe } from "vitest";
import { getVariantPerformances } from "./function";
import type { FunctionConfig } from "../config/function";
import type { MetricConfig } from "../config/metric";

describe("getVariantPerformances", () => {
  test("getVariantPerformances for dashboard_fixture_extract_entities", async () => {
    const function_name = "dashboard_fixture_extract_entities";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "dashboard_fixture_jaccard_similarity";
    const metric_config = {
      type: "float",
      level: "inference",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
    });
    expect(result).toEqual([
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 90,
        avg_metric: 0.4937301605939865,
        stdev: 0.4307567,
        ci_error: 0.08899525378450246,
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "llama_8b_initial_prompt",
        count: 110,
        avg_metric: 0.4099396590482105,
        stdev: 0.3624926,
        ci_error: 0.06774213240851094,
      },
    ]);
  });
  test("getVariantPerformances for dashboard_fixture_write_haiku", async () => {
    const function_name = "dashboard_fixture_write_haiku";
    const function_config = {
      type: "chat",
    } as FunctionConfig;
    const metric_name = "dashboard_fixture_haiku_rating";
    const metric_config = {
      type: "float",
      level: "inference",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
    });
    expect(result).toEqual([
      {
        period_start: "2024-12-23T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 491,
        avg_metric: 0.17349116723056418,
        stdev: 0.48264572,
        ci_error: 0.04269173301973782,
      },
    ]);
  });

  test("getVariantPerformances for ask_question and solved", async () => {
    const function_name = "ask_question";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "solved";
    const metric_config = {
      type: "boolean",
      level: "episode",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
    });
    expect(result).toEqual([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 23,
        avg_metric: 0.043478260869565216,
        stdev: 0.20851441405707477,
        ci_error: 0.08521739130434784,
      },
    ]);
  });

  test("getVariantPerformances for ask_question and num_questions", async () => {
    const function_name = "ask_question";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "num_questions";
    const metric_config = {
      type: "float",
      level: "episode",
      optimize: "min",
    } as MetricConfig;
    const time_window_unit = "week";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
    });
    expect(result).toEqual([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 49,
        avg_metric: 15.653061224489797,
        stdev: 5.9496174,
        ci_error: 1.665892868041992,
      },
    ]);
  });
});
