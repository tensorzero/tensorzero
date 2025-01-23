import {
  countInferencesForFunction,
  queryInferenceTableBoundsByFunctionName,
  queryInferenceTableByFunctionName,
} from "~/utils/clickhouse/inference";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  useSearchParams,
} from "react-router";
import { Badge } from "~/components/ui/badge";
import PageButtons from "~/components/utils/PageButtons";
import { getConfig } from "~/utils/config/index.server";
import FunctionInferenceTable from "./FunctionInferenceTable";
import BasicInfo from "./BasicInfo";
import { useConfig } from "~/context/config";
import {
  getVariantCounts,
  getVariantPerformances,
  type TimeWindowUnit,
} from "~/utils/clickhouse/function";
import { queryMetricsWithFeedback } from "~/utils/clickhouse/feedback";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { useState } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import FunctionVariantTable from "./FunctionVariantTable";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { function_name } = params;
  const url = new URL(request.url);
  const config = await getConfig();
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  const metric_name = url.searchParams.get("metric_name") || undefined;
  const time_granularity = url.searchParams.get("time_granularity") || "week";
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const function_config = config.functions[function_name];
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
    metrics: config.metrics,
  });
  const variantCountsPromise = getVariantCounts({
    function_name,
    function_config,
  });
  const variantPerformancesPromise = metric_name
    ? getVariantPerformances({
        function_name,
        function_config,
        metric_name,
        metric_config: config.metrics[metric_name],
        time_window_unit: time_granularity as TimeWindowUnit,
      })
    : undefined;

  const [
    inferences,
    inference_bounds,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_counts,
  ] = await Promise.all([
    inferencePromise,
    tableBoundsPromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
    variantCountsPromise,
  ]);
  const variant_counts_with_metadata = variant_counts.map((variant_count) => {
    const variant_config = function_config.variants[variant_count.variant_name];
    return {
      ...variant_count,
      type: variant_config.type,
      weight: variant_config.weight,
    };
  });
  return {
    function_name,
    inferences,
    inference_bounds,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
    variant_counts: variant_counts_with_metadata,
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
    variant_counts,
  } = loaderData;
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const function_config = useConfig().functions[function_name];

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

  const [metric_name, setMetricName] = useState(
    () => searchParams.get("metric_name") || "",
  );

  const handleMetricChange = (metric: string) => {
    setMetricName(metric);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("metric_name", metric);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const [time_granularity, setTimeGranularity] =
    useState<TimeWindowUnit>("week");
  const handleTimeGranularityChange = (granularity: TimeWindowUnit) => {
    setTimeGranularity(granularity);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("time_granularity", granularity);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="mb-4 text-2xl font-semibold">
        Function{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">
          {function_name}
        </code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <BasicInfo functionConfig={function_config} />
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <h3 className="mb-2 flex items-center gap-2 text-xl font-semibold">
        Variants
      </h3>
      <FunctionVariantTable
        variant_counts={variant_counts}
        function_name={function_name}
      />
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <MetricSelector
        metricsWithFeedback={metricsWithFeedback}
        selectedMetric={metric_name || ""}
        onMetricChange={handleMetricChange}
      />
      {variant_performances && (
        <div className="mt-6">
          <VariantPerformance
            variant_performances={variant_performances}
            metric_name={metric_name}
            time_granularity={time_granularity}
            onTimeGranularityChange={handleTimeGranularityChange}
          />
        </div>
      )}
      <div className="mt-6">
        <h3 className="mb-2 flex items-center gap-2 text-xl font-semibold">
          Inferences
          <Badge variant="secondary">Count: {num_inferences}</Badge>
        </h3>
        {inferences.length > 0 ? (
          <>
            <FunctionInferenceTable inferences={inferences} />
            <PageButtons
              onPreviousPage={handlePreviousInferencePage}
              onNextPage={handleNextInferencePage}
              disablePrevious={disablePreviousInferencePage}
              disableNext={disableNextInferencePage}
            />
          </>
        ) : (
          <div className="rounded-lg border border-gray-200 p-4 text-center text-gray-500">
            No inferences found
          </div>
        )}
      </div>
    </div>
  );
}
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

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
