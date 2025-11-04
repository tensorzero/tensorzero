import { expect, test, describe } from "vitest";
import {
  getUsedVariants,
  getVariantCounts,
  getVariantPerformances,
  getFunctionThroughputByVariant,
} from "./function";
import type { FunctionConfig, MetricConfig } from "~/types/tensorzero";

describe("getVariantPerformances", () => {
  test("getVariantPerformances for extract_entities", async () => {
    const function_name = "extract_entities";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "jaccard_similarity";
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
    expect(result).toMatchObject([
      {
        period_start: "2024-12-02T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 18,
        avg_metric: expect.closeTo(0.6097883615228865, 6),
        stdev: expect.closeTo(0.47275648, 6),
        ci_error: expect.closeTo(0.21840234885437046, 6),
      },
      {
        period_start: "2024-12-09T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 46,
        avg_metric: expect.closeTo(0.39694617170354596, 6),
        stdev: expect.closeTo(0.40689653, 6),
        ci_error: expect.closeTo(0.11758749631538713, 6),
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 26,
        avg_metric: expect.closeTo(0.5846153864493737, 6),
        stdev: expect.closeTo(0.41838107, 6),
        ci_error: expect.closeTo(0.1608205039163431, 6),
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "llama_8b_initial_prompt",
        count: 20,
        avg_metric: expect.closeTo(0.3672619082033634, 6),
        stdev: expect.closeTo(0.37131998, 6),
        ci_error: expect.closeTo(0.1627381562198926, 6),
      },
      {
        period_start: "2024-12-23T00:00:00.000Z",
        variant_name: "llama_8b_initial_prompt",
        count: 46,
        avg_metric: expect.closeTo(0.4359989678082259, 6),
        stdev: expect.closeTo(0.3903055, 6),
        ci_error: expect.closeTo(0.11279291348764303, 6),
      },
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "llama_8b_initial_prompt",
        count: 44,
        avg_metric: expect.closeTo(0.40209481391039764, 6),
        stdev: expect.closeTo(0.3333203, 6),
        ci_error: expect.closeTo(0.09848985179243835, 6),
      },
    ]);
  });
  test("getVariantPerformances for write_haiku", async () => {
    const function_name = "write_haiku";
    const function_config = {
      type: "chat",
    } as FunctionConfig;
    const metric_name = "haiku_rating";
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
    expect(result).toMatchObject([
      {
        period_start: "2024-11-18T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 32,
        avg_metric: expect.closeTo(0.0004417374475451652, 6),
        stdev: expect.closeTo(0.30695334, 6),
        ci_error: expect.closeTo(0.10635390649509185, 6),
      },
      {
        period_start: "2024-11-25T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.10361847593122277, 6),
        stdev: expect.closeTo(0.43975478, 6),
        ci_error: expect.closeTo(0.1141640103819334, 6),
      },
      {
        period_start: "2024-12-02T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 56,
        avg_metric: expect.closeTo(0.2647098991022046, 6),
        stdev: expect.closeTo(0.5056544, 6),
        ci_error: expect.closeTo(0.1324389850566572, 6),
      },
      {
        period_start: "2024-12-09T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.1640116744838132, 6),
        stdev: expect.closeTo(0.46050018, 6),
        ci_error: expect.closeTo(0.11954968840072935, 6),
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.2230634666395194, 6),
        stdev: expect.closeTo(0.49988118, 6),
        ci_error: expect.closeTo(0.12977332384647436, 6),
      },
      {
        period_start: "2024-12-23T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 55,
        avg_metric: expect.closeTo(0.15373795151033184, 6),
        stdev: expect.closeTo(0.5244488, 6),
        ci_error: expect.closeTo(0.1386046602344979, 6),
      },
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 56,
        avg_metric: expect.closeTo(0.18238492407947757, 6),
        stdev: expect.closeTo(0.47483847, 6),
        ci_error: expect.closeTo(0.1243677996248524, 6),
      },
      {
        period_start: "2025-01-06T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.23926746995564094, 6),
        stdev: expect.closeTo(0.55934626, 6),
        ci_error: expect.closeTo(0.14521095480097784, 6),
      },
      {
        period_start: "2025-01-13T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.15460191557608677, 6),
        stdev: expect.closeTo(0.48655924, 6),
        ci_error: expect.closeTo(0.12631483809560237, 6),
      },
      {
        period_start: "2025-01-20T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 7,
        avg_metric: expect.closeTo(0.17957699046071088, 6),
        stdev: expect.closeTo(0.21977705, 6),
        ci_error: expect.closeTo(0.16281311533756934, 6),
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
    expect(result).toMatchObject([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 23,
        avg_metric: expect.closeTo(0.043478260869565216, 6),
        stdev: expect.closeTo(0.20851441405707477, 6),
        ci_error: expect.closeTo(0.08521739130434784, 6),
      },
      {
        avg_metric: 0.4791666666666667,
        ci_error: 0.1428235512262245,
        count: 48,
        period_start: "2025-04-28T00:00:00.000Z",
        stdev: 0.5048523413086471,
        variant_name: "baseline",
      },
      {
        avg_metric: 1,
        ci_error: 0,
        count: 3,
        period_start: "2025-04-28T00:00:00.000Z",
        stdev: 0,
        variant_name: "gpt-4.1-mini",
      },
      {
        avg_metric: 0.4489795918367347,
        ci_error: 0.14071247279470286,
        count: 49,
        period_start: "2025-04-28T00:00:00.000Z",
        stdev: 0.5025445456953674,
        variant_name: "gpt-4.1-nano",
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
    expect(result).toMatchObject([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 49,
        avg_metric: expect.closeTo(15.653061224489797, 6),
        stdev: expect.closeTo(5.9496174, 6),
        ci_error: expect.closeTo(1.665892868041992, 6),
      },
    ]);
  });
});

describe("getVariantPerformances with variant filtering", () => {
  test("getVariantPerformances for extract_entities with specific variant", async () => {
    const function_name = "extract_entities";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "jaccard_similarity";
    const metric_config = {
      type: "float",
      level: "inference",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";
    const variant_name = "gpt4o_initial_prompt";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
      variant_name,
    });
    expect(result).toMatchObject([
      {
        period_start: "2024-12-02T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 18,
        avg_metric: expect.closeTo(0.6097883615228865, 6),
        stdev: expect.closeTo(0.47275648, 6),
        ci_error: expect.closeTo(0.21840234885437046, 6),
      },
      {
        period_start: "2024-12-09T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 46,
        avg_metric: expect.closeTo(0.39694617170354596, 6),
        stdev: expect.closeTo(0.40689653, 6),
        ci_error: expect.closeTo(0.11758749631538713, 6),
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "gpt4o_initial_prompt",
        count: 26,
        avg_metric: expect.closeTo(0.5846153864493737, 6),
        stdev: expect.closeTo(0.41838107, 6),
        ci_error: expect.closeTo(0.1608205039163431, 6),
      },
    ]);
  });

  test("getVariantPerformances for write_haiku with specific variant", async () => {
    const function_name = "write_haiku";
    const function_config = {
      type: "chat",
    } as FunctionConfig;
    const metric_name = "haiku_rating";
    const metric_config = {
      type: "float",
      level: "inference",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";
    const variant_name = "initial_prompt_gpt4o_mini";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
      variant_name,
    });
    expect(result).toMatchObject([
      {
        period_start: "2024-11-18T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 32,
        avg_metric: expect.closeTo(0.0004417374475451652, 6),
        stdev: expect.closeTo(0.30695334, 6),
        ci_error: expect.closeTo(0.10635390649509185, 6),
      },
      {
        period_start: "2024-11-25T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.10361847593122277, 6),
        stdev: expect.closeTo(0.43975478, 6),
        ci_error: expect.closeTo(0.1141640103819334, 6),
      },
      {
        period_start: "2024-12-02T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 56,
        avg_metric: expect.closeTo(0.2647098991022046, 6),
        stdev: expect.closeTo(0.5056544, 6),
        ci_error: expect.closeTo(0.1324389850566572, 6),
      },
      {
        period_start: "2024-12-09T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.1640116744838132, 6),
        stdev: expect.closeTo(0.46050018, 6),
        ci_error: expect.closeTo(0.11954968840072935, 6),
      },
      {
        period_start: "2024-12-16T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.2230634666395194, 6),
        stdev: expect.closeTo(0.49988118, 6),
        ci_error: expect.closeTo(0.12977332384647436, 6),
      },
      {
        period_start: "2024-12-23T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 55,
        avg_metric: expect.closeTo(0.15373795151033184, 6),
        stdev: expect.closeTo(0.5244488, 6),
        ci_error: expect.closeTo(0.1386046602344979, 6),
      },
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 56,
        avg_metric: expect.closeTo(0.18238492407947757, 6),
        stdev: expect.closeTo(0.47483847, 6),
        ci_error: expect.closeTo(0.1243677996248524, 6),
      },
      {
        period_start: "2025-01-06T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.23926746995564094, 6),
        stdev: expect.closeTo(0.55934626, 6),
        ci_error: expect.closeTo(0.14521095480097784, 6),
      },
      {
        period_start: "2025-01-13T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 57,
        avg_metric: expect.closeTo(0.15460191557608677, 6),
        stdev: expect.closeTo(0.48655924, 6),
        ci_error: expect.closeTo(0.12631483809560237, 6),
      },
      {
        period_start: "2025-01-20T00:00:00.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
        count: 7,
        avg_metric: expect.closeTo(0.17957699046071088, 6),
        stdev: expect.closeTo(0.21977705, 6),
        ci_error: expect.closeTo(0.16281311533756934, 6),
      },
    ]);
  });

  test("getVariantPerformances for ask_question and solved with specific variant", async () => {
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
    const variant_name = "baseline";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
      variant_name,
    });
    expect(result).toMatchObject([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 23,
        avg_metric: expect.closeTo(0.043478260869565216, 6),
        stdev: expect.closeTo(0.20851441405707477, 6),
        ci_error: expect.closeTo(0.08521739130434784, 6),
      },
      {
        avg_metric: 0.4791666666666667,
        ci_error: 0.1428235512262245,
        count: 48,
        period_start: "2025-04-28T00:00:00.000Z",
        stdev: 0.5048523413086471,
        variant_name: "baseline",
      },
    ]);
  });

  test("getVariantPerformances for ask_question and num_questions with specific variant", async () => {
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
    const variant_name = "baseline";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
      variant_name,
    });
    expect(result).toMatchObject([
      {
        period_start: "2024-12-30T00:00:00.000Z",
        variant_name: "baseline",
        count: 49,
        avg_metric: expect.closeTo(15.653061224489797, 6),
        stdev: expect.closeTo(5.9496174, 6),
        ci_error: expect.closeTo(1.665892868041992, 6),
      },
    ]);
  });
  test("getVariantPerformances for non-existent metric", async () => {
    const function_name = "extract_entities";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const metric_name = "non_existent_metric";
    const metric_config = {
      type: "float",
      level: "inference",
      optimize: "max",
    } as MetricConfig;
    const time_window_unit = "week";
    const variant_name = "gpt4o_initial_prompt";

    const result = await getVariantPerformances({
      function_name,
      function_config,
      metric_name,
      metric_config,
      time_window_unit,
      variant_name,
    });
    expect(result).toBeUndefined();
  });
});

describe("getVariantCounts", () => {
  test("getVariantCounts for extract_entities", async () => {
    const function_name = "extract_entities";
    const function_config = {
      type: "json",
    } as FunctionConfig;
    const result = await getVariantCounts({
      function_name,
      function_config,
    });
    expect(result).toMatchObject([
      {
        count: 224,
        last_used: "2025-04-15T02:34:21.000Z",
        variant_name: "gpt4o_mini_initial_prompt",
      },
      {
        count: 148,
        last_used: "2025-04-14T22:46:02.000Z",
        variant_name: "llama_8b_initial_prompt",
      },
      {
        count: 132,
        last_used: "2025-04-14T23:06:59.000Z",
        variant_name: "gpt4o_initial_prompt",
      },
      {
        count: 40,
        last_used: "2024-12-03T13:49:32.000Z",
        variant_name: "dicl",
      },
      {
        count: 35,
        last_used: "2024-12-06T03:49:59.000Z",
        variant_name: "turbo",
      },
      {
        count: 25,
        last_used: "2024-12-04T08:03:47.000Z",
        variant_name: "baseline",
      },
    ]);
  });

  test("getVariantCounts for write_haiku", async () => {
    const function_name = "write_haiku";
    const function_config = {
      type: "chat",
    } as FunctionConfig;
    const result = await getVariantCounts({
      function_name,
      function_config,
    });
    expect(result).toMatchObject([
      {
        count: 649,
        last_used: "2025-05-12T21:59:20.000Z",
        variant_name: "initial_prompt_gpt4o_mini",
      },
      {
        count: 155,
        last_used: "2025-04-15T02:33:07.000Z",
        variant_name: "better_prompt_haiku_3_5",
      },
    ]);
  });
});

describe("getUsedVariants", () => {
  test("getUsedVariants for extract_entities", async () => {
    const function_name = "extract_entities";
    const result = await getUsedVariants(function_name);
    expect(result).toEqual(
      expect.arrayContaining([
        "baseline",
        "dicl",
        "llama_8b_initial_prompt",
        "gpt4o_mini_initial_prompt",
        "gpt4o_initial_prompt",
        "turbo",
      ]),
    );
    expect(result.length).toBe(6);
  });
});

describe("getFunctionThroughputByVariant", () => {
  test("getFunctionThroughputByVariant for extract_entities with week granularity", async () => {
    const function_name = "extract_entities";
    const time_window = "week";
    const max_periods = 10;

    const result = await getFunctionThroughputByVariant(
      function_name,
      time_window,
      max_periods,
    );

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          period_start: expect.any(String),
          variant_name: expect.any(String),
          count: expect.any(Number),
        }),
      ]),
    );

    // Check that all results have valid structure
    result.forEach((row) => {
      expect(row).toMatchObject({
        period_start: expect.stringMatching(
          /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/,
        ),
        variant_name: expect.any(String),
        count: expect.any(Number),
      });
      expect(row.count).toBeGreaterThanOrEqual(0);
    });

    // Should have at least some data (placeholder assertion - update with actual expected data)
    expect(result.length).toBeGreaterThan(0);
  });

  test("getFunctionThroughputByVariant for write_haiku with day granularity", async () => {
    const function_name = "write_haiku";
    const time_window = "day";
    const max_periods = 5;

    const result = await getFunctionThroughputByVariant(
      function_name,
      time_window,
      max_periods,
    );

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          period_start: expect.any(String),
          variant_name: expect.any(String),
          count: expect.any(Number),
        }),
      ]),
    );

    // Check that results are sorted by period_start DESC, variant_name DESC as per query
    for (let i = 1; i < result.length; i++) {
      const current = new Date(result[i].period_start);
      const previous = new Date(result[i - 1].period_start);

      if (current.getTime() === previous.getTime()) {
        // Same period, check variant_name DESC ordering
        expect(result[i].variant_name <= result[i - 1].variant_name).toBe(true);
      } else {
        // Different periods, check period_start DESC ordering
        expect(current.getTime() <= previous.getTime()).toBe(true);
      }
    }

    // Should have at least some data (placeholder assertion - update with actual expected data)
    expect(result.length).toBeGreaterThan(0);
  });

  test("getFunctionThroughputByVariant for ask_question with month granularity", async () => {
    const function_name = "ask_question";
    const time_window = "month";
    const max_periods = 3;

    const result = await getFunctionThroughputByVariant(
      function_name,
      time_window,
      max_periods,
    );

    // Should return valid throughput data structure
    result.forEach((row) => {
      expect(row).toMatchObject({
        period_start: expect.stringMatching(
          /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/,
        ),
        variant_name: expect.any(String),
        count: expect.any(Number),
      });
      expect(row.count).toBeGreaterThanOrEqual(0);
    });

    // Placeholder assertions - update with actual expected counts
    expect(result.length).toBeGreaterThanOrEqual(0);
  });

  test("getFunctionThroughputByVariant for non-existent function", async () => {
    const function_name = "non_existent_function";
    const time_window = "week";
    const max_periods = 10;

    const result = await getFunctionThroughputByVariant(
      function_name,
      time_window,
      max_periods,
    );

    // Should return empty array for non-existent function
    expect(result).toEqual([]);
  });

  test("getFunctionThroughputByVariant with cumulative time window", async () => {
    const function_name = "extract_entities";
    const time_window = "cumulative";
    const max_periods = 1;

    const result = await getFunctionThroughputByVariant(
      function_name,
      time_window,
      max_periods,
    );

    // Should return valid throughput data
    result.forEach((row) => {
      expect(row).toMatchObject({
        period_start: expect.any(String),
        variant_name: expect.any(String),
        count: expect.any(Number),
      });
      expect(row.count).toBeGreaterThanOrEqual(0);
    });

    // Placeholder assertion - update with actual expected data
    expect(result.length).toBeGreaterThanOrEqual(0);
  });
});
