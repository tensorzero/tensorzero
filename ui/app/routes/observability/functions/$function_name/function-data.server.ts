import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";
import type { FunctionConfig, TimeWindow } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type { getConfig } from "~/utils/config/index.server";

/**
 * Data for the Inferences section
 */
export type InferencesSectionData = {
  inferences: Awaited<
    ReturnType<ReturnType<typeof getTensorZeroClient>["listInferenceMetadata"]>
  >["inference_metadata"];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  count: number;
};

/**
 * Data for the Variants section
 */
export type VariantsSectionData = {
  variant_counts: {
    variant_name: string;
    inference_count: bigint;
    last_used_at: string;
    type: string;
    weight: number | null;
  }[];
};

/**
 * Data for the Throughput section
 */
export type ThroughputSectionData = Awaited<
  ReturnType<
    ReturnType<typeof getTensorZeroClient>["getFunctionThroughputByVariant"]
  >
>["throughput"];

/**
 * Data for the Metrics section
 */
export type MetricsSectionData = {
  metricsWithFeedback: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getFunctionMetricsWithFeedback"]
    >
  >;
  variant_performances:
    | Awaited<
        ReturnType<
          ReturnType<typeof getTensorZeroClient>["getVariantPerformances"]
        >
      >["performances"]
    | undefined;
};

/**
 * Data for the Experimentation section
 */
export type ExperimentationSectionData = {
  feedback_timeseries:
    | Awaited<
        ReturnType<
          ReturnType<
            typeof getTensorZeroClient
          >["getCumulativeFeedbackTimeseries"]
        >
      >
    | undefined;
  variant_sampling_probabilities: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getVariantSamplingProbabilities"]
    >
  >["probabilities"];
};

type FetchInferencesParams = {
  function_name: string;
  beforeInference: string | null;
  afterInference: string | null;
  limit: number;
};

export async function fetchInferencesSectionData(
  params: FetchInferencesParams,
): Promise<InferencesSectionData> {
  const { function_name, beforeInference, afterInference, limit } = params;

  const client = getTensorZeroClient();
  const [inferenceResult, count] = await Promise.all([
    client.listInferenceMetadata({
      function_name,
      before: beforeInference || undefined,
      after: afterInference || undefined,
      limit: limit + 1, // Fetch one extra to determine pagination
    }),
    countInferencesForFunction(function_name),
  ]);

  const { items, hasNextPage, hasPreviousPage } = applyPaginationLogic(
    inferenceResult.inference_metadata,
    limit,
    {
      before: beforeInference,
      after: afterInference,
    },
  );

  return {
    inferences: items,
    hasNextPage,
    hasPreviousPage,
    count,
  };
}

type FetchVariantsParams = {
  function_name: string;
  function_config: FunctionConfig;
};

export async function fetchVariantsSectionData(
  params: FetchVariantsParams,
): Promise<VariantsSectionData> {
  const { function_name, function_config } = params;

  const client = getTensorZeroClient();
  const variant_counts = await client.getInferenceCount(function_name, {
    groupBy: "variant",
  });

  const variant_counts_with_metadata = (
    variant_counts.count_by_variant ?? []
  ).map((variant_count) => {
    let variant_config = function_config.variants[
      variant_count.variant_name
    ] || {
      inner: {
        type: "unknown",
        weight: 0,
      },
    };

    if (function_name === DEFAULT_FUNCTION) {
      variant_config = {
        inner: {
          type: "chat_completion",
          model: variant_count.variant_name,
          weight: null,
          templates: {},
          temperature: null,
          top_p: null,
          max_tokens: null,
          presence_penalty: null,
          frequency_penalty: null,
          seed: null,
          stop_sequences: null,
          json_mode: null,
          retries: { num_retries: 0, max_delay_s: 0 },
        },
        timeouts: {
          non_streaming: { total_ms: null },
          streaming: { ttft_ms: null },
        },
      };
    }

    return {
      ...variant_count,
      type: variant_config.inner.type,
      weight: variant_config.inner.weight,
    };
  });

  return { variant_counts: variant_counts_with_metadata };
}

type FetchThroughputParams = {
  function_name: string;
  time_granularity: TimeWindow;
};

export async function fetchThroughputSectionData(
  params: FetchThroughputParams,
): Promise<ThroughputSectionData> {
  const { function_name, time_granularity } = params;

  const client = getTensorZeroClient();
  const response = await client.getFunctionThroughputByVariant(
    function_name,
    time_granularity,
    10,
  );

  return response.throughput;
}

type FetchMetricsParams = {
  function_name: string;
  metric_name: string | undefined;
  time_granularity: TimeWindow;
  config: Awaited<ReturnType<typeof getConfig>>;
};

export async function fetchMetricsSectionData(
  params: FetchMetricsParams,
): Promise<MetricsSectionData> {
  const { function_name, metric_name, time_granularity, config } = params;

  const client = getTensorZeroClient();

  const [metricsWithFeedback, variant_performances] = await Promise.all([
    client.getFunctionMetricsWithFeedback(function_name),
    metric_name && config.metrics[metric_name]
      ? client
          .getVariantPerformances(function_name, metric_name, time_granularity)
          .then((response) =>
            response.performances.length > 0
              ? response.performances
              : undefined,
          )
      : Promise.resolve(undefined),
  ]);

  return { metricsWithFeedback, variant_performances };
}

type FetchExperimentationParams = {
  function_name: string;
  function_config: FunctionConfig;
  time_granularity: TimeWindow;
};

export async function fetchExperimentationSectionData(
  params: FetchExperimentationParams,
): Promise<ExperimentationSectionData> {
  const { function_name, function_config, time_granularity } = params;

  const client = getTensorZeroClient();

  const feedbackParams =
    function_config.experimentation.type === "track_and_stop"
      ? {
          metric_name: function_config.experimentation.metric,
          variant_names: function_config.experimentation.candidate_variants,
        }
      : null;

  const [feedback_timeseries, variant_sampling_probabilities] =
    await Promise.all([
      feedbackParams
        ? client.getCumulativeFeedbackTimeseries({
            function_name,
            ...feedbackParams,
            time_window: time_granularity,
            max_periods: 10,
          })
        : Promise.resolve(undefined),
      client
        .getVariantSamplingProbabilities(function_name)
        .then((response) => response.probabilities),
    ]);

  return { feedback_timeseries, variant_sampling_probabilities };
}
