import { expect, test, describe } from "vitest";
import type { VariantPerformanceRow } from "~/types/tensorzero";
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
        ci_error: 0.08899525378450246,
      },
      {
        period_start: "2024-12-16",
        variant_name: "llama_8b_initial_prompt",
        count: 110,
        avg_metric: 0.4099396590482105,
        stdev: 0.3624926,
        ci_error: 0.06774213240851094,
      },
    ];
    const { data, variantNames } = transformVariantPerformances(input);
    expect(data).toEqual([
      {
        date: "2024-12-16",
        gpt4o_initial_prompt: 0.4937301605939865,
        gpt4o_initial_prompt_ci_error: 0.08899525378450246,
        gpt4o_initial_prompt_num_inferences: 90,
        llama_8b_initial_prompt: 0.4099396590482105,
        llama_8b_initial_prompt_ci_error: 0.06774213240851094,
        llama_8b_initial_prompt_num_inferences: 110,
      },
    ]);
    expect(variantNames).toEqual([
      "gpt4o_initial_prompt",
      "llama_8b_initial_prompt",
    ]);
  });

  test("transforms single period with single variant", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-23",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 491,
        avg_metric: 0.17349116723056418,
        stdev: 0.48264572,
        ci_error: 0.08521739130434784,
      },
    ];
    const { data, variantNames } = transformVariantPerformances(input);
    expect(data).toEqual([
      {
        date: "2024-12-23",
        initial_prompt_gpt4o_mini: 0.17349116723056418,
        initial_prompt_gpt4o_mini_ci_error: 0.08521739130434784,
        initial_prompt_gpt4o_mini_num_inferences: 491,
      },
    ]);
    expect(variantNames).toEqual(["initial_prompt_gpt4o_mini"]);
  });

  test("transforms single variant with episode level metrics", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-30",
        variant_name: "baseline",
        count: 23,
        avg_metric: 0.043478260869565216,
        stdev: 0.20851441405707477,
        ci_error: 0.08521739130434784,
      },
    ];

    const { data, variantNames } = transformVariantPerformances(input);
    expect(data).toEqual([
      {
        date: "2024-12-30",
        baseline: 0.043478260869565216,
        baseline_ci_error: 0.08521739130434784,
        baseline_num_inferences: 23,
      },
    ]);
    expect(variantNames).toEqual(["baseline"]);
  });

  test("transforms multiple periods with multiple variants", () => {
    const input: VariantPerformanceRow[] = [
      {
        period_start: "2024-12-16",
        variant_name: "gpt4o_initial_prompt",
        count: 90,
        avg_metric: 0.49,
        stdev: 0.43,
        ci_error: 0.08899525378450246,
      },
      {
        period_start: "2024-12-16",
        variant_name: "llama_8b_initial_prompt",
        count: 110,
        avg_metric: 0.41,
        stdev: 0.36,
        ci_error: 0.06774213240851094,
      },
      {
        period_start: "2024-12-23",
        variant_name: "gpt4o_initial_prompt",
        count: 95,
        avg_metric: 0.51,
        stdev: 0.44,
        ci_error: 0.08899525378450246,
      },
      {
        period_start: "2024-12-23",
        variant_name: "llama_8b_initial_prompt",
        count: 105,
        avg_metric: 0.43,
        stdev: 0.38,
        ci_error: 0.06774213240851094,
      },
    ];

    const { data, variantNames } = transformVariantPerformances(input);
    expect(data).toEqual([
      {
        date: "2024-12-16",
        gpt4o_initial_prompt: 0.49,
        gpt4o_initial_prompt_ci_error: 0.08899525378450246,
        gpt4o_initial_prompt_num_inferences: 90,
        llama_8b_initial_prompt: 0.41,
        llama_8b_initial_prompt_ci_error: 0.06774213240851094,
        llama_8b_initial_prompt_num_inferences: 110,
      },
      {
        date: "2024-12-23",
        gpt4o_initial_prompt: 0.51,
        gpt4o_initial_prompt_ci_error: 0.08899525378450246,
        gpt4o_initial_prompt_num_inferences: 95,
        llama_8b_initial_prompt: 0.43,
        llama_8b_initial_prompt_ci_error: 0.06774213240851094,
        llama_8b_initial_prompt_num_inferences: 105,
      },
    ]);
    expect(variantNames).toEqual([
      "gpt4o_initial_prompt",
      "llama_8b_initial_prompt",
    ]);
  });
});
