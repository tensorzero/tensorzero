import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import { data, useNavigate, useSearchParams } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import FunctionInferenceTable from "./FunctionInferenceTable";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
import { FunctionExperimentation } from "./FunctionExperimentation";
import { useFunctionConfig } from "~/context/config";
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
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { FunctionTypeBadge } from "~/components/function/FunctionSelector";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type { TimeWindow } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { function_name } = params;
  const url = new URL(request.url);
  const config = await getConfig();
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const limit = Number(url.searchParams.get("limit")) || 10;
  const metric_name = url.searchParams.get("metric_name") || undefined;
  const time_granularity = (url.searchParams.get("time_granularity") ||
    "week") as TimeWindow;
  const throughput_time_granularity = (url.searchParams.get(
    "throughput_time_granularity",
  ) || "week") as TimeWindow;
  const feedback_time_granularity = (url.searchParams.get(
    "cumulative_feedback_time_granularity",
  ) || "week") as TimeWindow;
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const function_config = await getFunctionConfig(function_name, config);
  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }
  const client = getTensorZeroClient();
  const inferencePromise = client.listInferenceMetadata({
    function_name,
    before: beforeInference || undefined,
    after: afterInference || undefined,
    limit: limit + 1, // Fetch one extra to determine pagination
  });
  const numInferencesPromise = countInferencesForFunction(function_name);
  const tensorZeroClient = getTensorZeroClient();
  const metricsWithFeedbackPromise =
    tensorZeroClient.getFunctionMetricsWithFeedback(function_name);
  const variantCountsPromise = tensorZeroClient.getInferenceCount(
    function_name,
    {
      groupBy: "variant",
    },
  );
  const variantPerformancesPromise =
    // Only get variant performances if metric_name is provided and valid
    metric_name && config.metrics[metric_name]
      ? tensorZeroClient
          .getVariantPerformances(function_name, metric_name, time_granularity)
          .then((response) =>
            response.performances.length > 0
              ? response.performances
              : undefined,
          )
      : Promise.resolve(undefined);
  const variantThroughputPromise = tensorZeroClient
    .getFunctionThroughputByVariant(
      function_name,
      throughput_time_granularity,
      10,
    )
    .then((response) => response.throughput);

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
    ? tensorZeroClient.getCumulativeFeedbackTimeseries({
        function_name,
        ...feedbackParams,
        time_window: feedback_time_granularity as TimeWindow,
        max_periods: 10,
      })
    : Promise.resolve(undefined);

  // Get variant sampling probabilities from the gateway
  const variantSamplingProbabilitiesPromise = tensorZeroClient
    .getVariantSamplingProbabilities(function_name)
    .then((response) => response.probabilities);

  const [
    inferenceResult,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_counts,
    variant_throughput,
    feedback_timeseries,
    variant_sampling_probabilities,
  ] = await Promise.all([
    inferencePromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
    variantCountsPromise,
    variantThroughputPromise,
    feedbackTimeseriesPromise,
    variantSamplingProbabilitiesPromise,
  ]);

  const variant_counts_with_metadata = (
    variant_counts.count_by_variant ?? []
  ).map((variant_count) => {
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

  // Add config variants that have no inferences yet (not applicable to DEFAULT_FUNCTION)
  if (function_name !== DEFAULT_FUNCTION) {
    const existingVariants = new Set(
      variant_counts_with_metadata.map((v) => v.variant_name),
    );
    const missingVariants = Object.entries(function_config.variants)
      .filter(([name]) => !existingVariants.has(name))
      .map(([name, config]) => ({
        variant_name: name,
        inference_count: 0,
        last_used_at: null,
        type: config.inner.type,
        weight: config.inner.weight,
      }));
    variant_counts_with_metadata.push(...missingVariants);
  }

  // Handle pagination from listInferenceMetadata response
  const {
    items: inferences,
    hasNextPage: hasNextInferencePage,
    hasPreviousPage: hasPreviousInferencePage,
  } = applyPaginationLogic(inferenceResult.inference_metadata, limit, {
    before: beforeInference,
    after: afterInference,
  });

  return {
    function_name,
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_throughput,
    variant_counts: variant_counts_with_metadata,
    feedback_timeseries,
    variant_sampling_probabilities,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    function_name,
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_throughput,
    variant_counts,
    feedback_timeseries,
    variant_sampling_probabilities,
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
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Functions", href: "/observability/functions" },
            ]}
          />
        }
        name={function_name}
        tag={<FunctionTypeBadge type={function_config.type} />}
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
              feedbackTimeseries={feedback_timeseries}
              variantSamplingProbabilities={variant_sampling_probabilities}
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
            disablePrevious={!hasPreviousInferencePage}
            disableNext={!hasNextInferencePage}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
