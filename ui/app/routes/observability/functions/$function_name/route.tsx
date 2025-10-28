import {
  countInferencesForFunction,
  queryInferenceTableBoundsByFunctionName,
  queryInferenceTableByFunctionName,
} from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  useSearchParams,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import FunctionInferenceTable from "./FunctionInferenceTable";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
import { FunctionExperimentation } from "./FunctionExperimentation";
import { useFunctionConfig } from "~/context/config";
import {
  getVariantCounts,
  getVariantPerformances,
  getFunctionThroughputByVariant,
} from "~/utils/clickhouse/function";
import { queryMetricsWithFeedback } from "~/utils/clickhouse/feedback";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { useMemo } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import { VariantThroughput } from "~/components/function/variant/VariantThroughput";
import FunctionVariantTable from "./FunctionVariantTable";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { getFunctionTypeIcon } from "~/utils/icon";
import { logger } from "~/utils/logger";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { TimeWindow, TrackAndStopResponse } from "tensorzero-node";
import { computeTrackAndStopState } from "tensorzero-node";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { function_name } = params;
  const url = new URL(request.url);
  const config = await getConfig();
  const dbClient = await getNativeDatabaseClient();
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  const metric_name = url.searchParams.get("metric_name") || undefined;
  const time_granularity = (url.searchParams.get("time_granularity") ||
    "week") as TimeWindow;
  const throughput_time_granularity = (url.searchParams.get(
    "throughput_time_granularity",
  ) || "week") as TimeWindow;
  const feedback_time_granularity = (url.searchParams.get(
    "cumulative_feedback_time_granularity",
  ) || "week") as TimeWindow;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const function_config = await getFunctionConfig(function_name, config);
  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }
  const inferencePromise = queryInferenceTableByFunctionName({
    function_name,
    before: beforeInference || undefined,
    after: afterInference || undefined,
    page_size: pageSize,
  });
  const tableBoundsPromise = queryInferenceTableBoundsByFunctionName({
    function_name,
  });
  const numInferencesPromise = countInferencesForFunction(
    function_name,
    function_config,
  );
  const metricsWithFeedbackPromise = queryMetricsWithFeedback({
    function_name,
    inference_table: getInferenceTableName(function_config),
  });
  const variantCountsPromise = getVariantCounts({
    function_name,
    function_config,
  });
  const variantPerformancesPromise =
    // Only get variant performances if metric_name is provided and valid
    metric_name && config.metrics[metric_name]
      ? getVariantPerformances({
          function_name,
          function_config,
          metric_name,
          metric_config: config.metrics[metric_name],
          time_window_unit: time_granularity,
        })
      : undefined;
  const variantThroughputPromise = getFunctionThroughputByVariant(
    function_name,
    throughput_time_granularity,
    10,
  );

  // Get feedback timeseries
  // For now, we only fetch this for track_and_stop experimentation
  // but the underlying query is general and could be used for other experimentation types
  const feedbackParams =
    function_config.experimentation.type === "track_and_stop"
      ? {
          metric_name: function_config.experimentation.metric,
          variant_names: function_config.experimentation.candidate_variants,
        }
      : null;
  const feedbackTimeseriesPromise = feedbackParams
    ? (async () => {
        return dbClient.getCumulativeFeedbackTimeseries({
          function_name,
          ...feedbackParams,
          time_window: feedback_time_granularity as TimeWindow,
          max_periods: 10,
        });
      })()
    : Promise.resolve(undefined);

  const [
    inferences,
    inference_bounds,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_counts,
    variant_throughput,
    feedback_timeseries,
  ] = await Promise.all([
    inferencePromise,
    tableBoundsPromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
    variantCountsPromise,
    variantThroughputPromise,
    feedbackTimeseriesPromise,
  ]);

  // Compute track-and-stop state if needed
  let track_and_stop_state: TrackAndStopResponse | undefined = undefined;

  if (function_config.experimentation.type === "track_and_stop") {
    try {
      const dbClient = await getNativeDatabaseClient();
      const experimentationConfig = function_config.experimentation;
      const metric_config = config.metrics[experimentationConfig.metric];

      if (metric_config) {
        const feedback = await dbClient.getFeedbackByVariant({
          metric_name: experimentationConfig.metric,
          function_name,
          variant_names: experimentationConfig.candidate_variants,
        });

        // Compute the track-and-stop state (includes sampling probabilities)
        // Convert BigInt fields to numbers before JSON serialization
        const stateJson = computeTrackAndStopState(
          JSON.stringify({
            candidate_variants: experimentationConfig.candidate_variants,
            feedback: feedback.map((f) => ({
              ...f,
              count: Number(f.count),
            })),
            min_samples_per_variant: Number(
              experimentationConfig.min_samples_per_variant,
            ),
            delta: experimentationConfig.delta,
            epsilon: experimentationConfig.epsilon,
            min_prob: experimentationConfig.min_prob,
            metric_optimize: metric_config.optimize,
          }),
        );

        track_and_stop_state = JSON.parse(stateJson);
      }
    } catch (error) {
      logger.error("Failed to compute track-and-stop state:", error);
    }
  }

  const variant_counts_with_metadata = variant_counts.map((variant_count) => {
    let variant_config = function_config.variants[
      variant_count.variant_name
    ] || {
      inner: {
        // In case the variant is not found, we still want to display the variant name
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
      function_config.variants[variant_count.variant_name] = variant_config;
    }

    return {
      ...variant_count,
      type: variant_config.inner.type,
      weight: variant_config.inner.weight,
    };
  });

  return {
    function_name,
    inferences,
    inference_bounds,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_throughput,
    variant_counts: variant_counts_with_metadata,
    track_and_stop_state,
    feedback_timeseries,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    function_name,
    inferences,
    inference_bounds,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_throughput,
    variant_counts,
    track_and_stop_state,
    feedback_timeseries,
  } = loaderData;

  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const function_config = useFunctionConfig(function_name);
  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }

  // Only get top/bottom inferences if array is not empty
  const topInference = inferences.length > 0 ? inferences[0] : null;
  const bottomInference =
    inferences.length > 0 ? inferences[inferences.length - 1] : null;

  const handleNextInferencePage = () => {
    if (!bottomInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  // Modify pagination disable logic to handle empty inferences
  const disablePreviousInferencePage =
    !topInference || inference_bounds.last_id === topInference.id;
  const disableNextInferencePage =
    !bottomInference || inference_bounds.first_id === bottomInference.id;

  const metric_name = searchParams.get("metric_name") || "";

  const handleMetricChange = (metric: string) => {
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("metric_name", metric);
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const metricsExcludingDemonstrations = useMemo(
    () => ({
      metrics: metricsWithFeedback.metrics.filter(
        ({ metric_type }) => metric_type !== "demonstration",
      ),
    }),
    [metricsWithFeedback],
  );

  return (
    <PageLayout>
      <PageHeader
        name={function_name}
        label={`${function_config.type} · Function`}
        icon={getFunctionTypeIcon(function_config.type).icon}
        iconBg={getFunctionTypeIcon(function_config.type).iconBg}
      >
        <BasicInfo functionConfig={function_config} />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Variants" />
          <FunctionVariantTable
            variant_counts={variant_counts}
            function_name={function_name}
          />
        </SectionLayout>

        {function_name !== DEFAULT_FUNCTION && (
          <SectionLayout>
            <SectionHeader heading="Experimentation" />
            <FunctionExperimentation
              functionConfig={function_config}
              functionName={function_name}
              trackAndStopState={track_and_stop_state}
              feedbackTimeseries={feedback_timeseries}
            />
          </SectionLayout>
        )}

        <SectionLayout>
          <SectionHeader heading="Throughput" />
          <VariantThroughput variant_throughput={variant_throughput} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Metrics" />
          <MetricSelector
            metricsWithFeedback={metricsExcludingDemonstrations}
            selectedMetric={metric_name || ""}
            onMetricChange={handleMetricChange}
          />
          {variant_performances && (
            <VariantPerformance
              variant_performances={variant_performances}
              metric_name={metric_name}
            />
          )}
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Schemas" />
          <FunctionSchema functionConfig={function_config} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Inferences" count={num_inferences} />
          <FunctionInferenceTable inferences={inferences} />
          <PageButtons
            onPreviousPage={handlePreviousInferencePage}
            onNextPage={handleNextInferencePage}
            disablePrevious={disablePreviousInferencePage}
            disableNext={disableNextInferencePage}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
