import { expect, test, describe } from "vitest";
import type { VariantPerformanceRow } from "~/utils/clickhouse/function";
import { transformVariantPerformances } from "./VariantPerformance";

describe("transformVariantPerformances", () => {
  test("transforms single period with multiple variants", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-16",
        variant_name: "gpt4o_initial_prompt",
        count: 90,
        avg_metric: 0.4937301605939865,
        stdev: 0.4307567,
        ci_lower_95: 0.40473490680948404,
        ci_upper_95: 0.582725414378489,
      },
      {
        period_start: "2024-12-16",
        variant_name: "llama_8b_initial_prompt",
        count: 110,
        avg_metric: 0.4099396590482105,
        stdev: 0.3624926,
        ci_lower_95: 0.34219752663969955,
        ci_upper_95: 0.47768179145672146,
      },
    ];

    const expected = [
      {
        date: "2024-12-16",
        variants: {
          gpt4o_initial_prompt: {
            num_inferences: 90,
            avg_metric: 0.4937301605939865,
            stdev: 0.4307567,
            ci_lower_95: 0.40473490680948404,
            ci_upper_95: 0.582725414378489,
          },
          llama_8b_initial_prompt: {
            num_inferences: 110,
            avg_metric: 0.4099396590482105,
            stdev: 0.3624926,
            ci_lower_95: 0.34219752663969955,
            ci_upper_95: 0.47768179145672146,
          },
        },
      },
    ];

    expect(transformVariantPerformances(input)).toEqual(expected);
  });

  test("transforms single period with single variant", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-23",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 491,
        avg_metric: 0.17349116723056418,
        stdev: 0.48264572,
        ci_lower_95: 0.13079943421082635,
        ci_upper_95: 0.216182900250302,
      },
    ];

    const expected = [
      {
        date: "2024-12-23",
        variants: {
          initial_prompt_gpt4o_mini: {
            num_inferences: 491,
            avg_metric: 0.17349116723056418,
            stdev: 0.48264572,
            ci_lower_95: 0.13079943421082635,
            ci_upper_95: 0.216182900250302,
          },
        },
      },
    ];

    expect(transformVariantPerformances(input)).toEqual(expected);
  });

  test("transforms single variant with episode level metrics", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-30",
        variant_name: "baseline",
        count: 23,
        avg_metric: 0.043478260869565216,
        stdev: 0.20851441405707477,
        ci_lower_95: -0.041739130434782626,
        ci_upper_95: 0.12869565217391304,
      },
    ];

    const expected = [
      {
        date: "2024-12-30",
        variants: {
          baseline: {
            num_inferences: 23,
            avg_metric: 0.043478260869565216,
            stdev: 0.20851441405707477,
            ci_lower_95: -0.041739130434782626,
            ci_upper_95: 0.12869565217391304,
          },
        },
      },
    ];

    expect(transformVariantPerformances(input)).toEqual(expected);
  });

  test("transforms multiple periods with multiple variants", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-16",
        variant_name: "gpt4o_initial_prompt",
        count: 90,
        avg_metric: 0.49,
        stdev: 0.43,
        ci_lower_95: 0.4,
        ci_upper_95: 0.58,
      },
      {
        period_start: "2024-12-16",
        variant_name: "llama_8b_initial_prompt",
        count: 110,
        avg_metric: 0.41,
        stdev: 0.36,
        ci_lower_95: 0.34,
        ci_upper_95: 0.48,
      },
      {
        period_start: "2024-12-23",
        variant_name: "gpt4o_initial_prompt",
        count: 95,
        avg_metric: 0.51,
        stdev: 0.44,
        ci_lower_95: 0.42,
        ci_upper_95: 0.6,
      },
      {
        period_start: "2024-12-23",
        variant_name: "llama_8b_initial_prompt",
        count: 105,
        avg_metric: 0.43,
        stdev: 0.38,
        ci_lower_95: 0.36,
        ci_upper_95: 0.5,
      },
    ];

    const expected = [
      {
        date: "2024-12-16",
        variants: {
          gpt4o_initial_prompt: {
            num_inferences: 90,
            avg_metric: 0.49,
            stdev: 0.43,
            ci_lower_95: 0.4,
            ci_upper_95: 0.58,
          },
          llama_8b_initial_prompt: {
            num_inferences: 110,
            avg_metric: 0.41,
            stdev: 0.36,
            ci_lower_95: 0.34,
            ci_upper_95: 0.48,
          },
        },
      },
      {
        date: "2024-12-23",
        variants: {
          gpt4o_initial_prompt: {
            num_inferences: 95,
            avg_metric: 0.51,
            stdev: 0.44,
            ci_lower_95: 0.42,
            ci_upper_95: 0.6,
          },
          llama_8b_initial_prompt: {
            num_inferences: 105,
            avg_metric: 0.43,
            stdev: 0.38,
            ci_lower_95: 0.36,
            ci_upper_95: 0.5,
          },
        },
      },
    ];

    expect(transformVariantPerformances(input)).toEqual(expected);
  });
});
