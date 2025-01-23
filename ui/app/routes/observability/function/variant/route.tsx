import {
  data,
  isRouteErrorResponse,
  redirect,
  useLoaderData,
  useNavigate,
  useSearchParams,
} from "react-router";
import type { LoaderFunctionArgs } from "react-router";
import BasicInfo from "./BasicInfo";
import { useConfig } from "~/context/config";
import PageButtons from "~/components/utils/PageButtons";
import { Badge } from "~/components/ui/badge";
import VariantInferenceTable from "./VariantInferenceTable";
import { getConfig } from "~/utils/config/index.server";
import {
  countInferencesForVariant,
  queryInferenceTableBoundsByVariantName,
  queryInferenceTableByVariantName,
} from "~/utils/clickhouse/inference";
import {
  getVariantPerformances,
  type TimeWindowUnit,
} from "~/utils/clickhouse/function";
import { useState } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import { queryMetricsWithFeedback } from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";

export async function loader({ request, params }: LoaderFunctionArgs) {
  const { function_name, variant_name } = params;
  if (!function_name || !variant_name) {
    return redirect("/observability/functions");
  }
  const config = await getConfig();
  const url = new URL(request.url);
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

  const inferencePromise = queryInferenceTableByVariantName({
    function_name,
    variant_name,
    page_size: pageSize,
    before: beforeInference || undefined,
    after: afterInference || undefined,
  });

  const tableBoundsPromise = queryInferenceTableBoundsByVariantName({
    function_name,
    variant_name,
  });

  const numInferencesPromise = countInferencesForVariant(
    function_name,
    function_config,
    variant_name,
  );
  const metricsWithFeedbackPromise = queryMetricsWithFeedback({
    function_name,
    inference_table: getInferenceTableName(function_config),
    metrics: config.metrics,
    variant_name,
  });

  const variantPerformancesPromise = metric_name
    ? getVariantPerformances({
        function_name,
        function_config,
        metric_name,
        metric_config: config.metrics[metric_name],
        time_window_unit: time_granularity as TimeWindowUnit,
        variant_name,
      })
    : undefined;

  const [
    inferences,
    inference_bounds,
    num_inferences,
    variant_performances,
    metricsWithFeedback,
  ] = await Promise.all([
    inferencePromise,
    tableBoundsPromise,
    numInferencesPromise,
    variantPerformancesPromise,
    metricsWithFeedbackPromise,
  ]);

  return {
    function_name,
    variant_name,
    num_inferences,
    inferences,
    inference_bounds,
    variant_performances,
    metricsWithFeedback,
  };
}

export default function VariantDetails() {
  const {
    function_name,
    variant_name,
    num_inferences,
    inferences,
    inference_bounds,
    variant_performances,
    metricsWithFeedback,
  } = useLoaderData<typeof loader>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const config = useConfig();
  const function_config = config.functions[function_name];
  if (!function_config) {
    throw new Response(
      "Function not found. This likely means there is data in ClickHouse from an old TensorZero config.",
      {
        status: 404,
        statusText: "Not Found",
      },
    );
  }
  const variant_config = function_config.variants[variant_name];
  if (!variant_config) {
    throw new Response(
      "Variant not found. This likely means there is data in ClickHouse from an old TensorZero config.",
      {
        status: 404,
        statusText: "Not Found",
      },
    );
  }

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
        Variant{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{variant_name}</code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <BasicInfo variantConfig={variant_config} function_name={function_name} />
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
            <VariantInferenceTable inferences={inferences} />
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
