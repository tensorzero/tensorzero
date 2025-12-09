import {
  data,
  isRouteErrorResponse,
  redirect,
  useNavigate,
  useSearchParams,
} from "react-router";
import type { LoaderFunctionArgs, RouteHandle } from "react-router";
import BasicInfo from "./VariantBasicInfo";
import VariantTemplate from "./VariantTemplate";
import { useFunctionConfig } from "~/context/config";
import PageButtons from "~/components/utils/PageButtons";
import VariantInferenceTable from "./VariantInferenceTable";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import {
  countInferencesForVariant,
  listInferencesWithPagination,
} from "~/utils/clickhouse/inference.server";
import { getVariantPerformances } from "~/utils/clickhouse/function";
import type { TimeWindow } from "~/types/tensorzero";
import { useMemo, useState } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import { queryMetricsWithFeedback } from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Variants",
    { label: match.params.variant_name!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: LoaderFunctionArgs) {
  const { function_name, variant_name } = params;
  if (!function_name || !variant_name) {
    return redirect("/observability/functions");
  }
  const config = await getConfig();
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const limit = Number(url.searchParams.get("limit")) || 10;
  const metric_name = url.searchParams.get("metric_name") || undefined;
  const time_granularity = url.searchParams.get("time_granularity") || "week";
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }
  const function_config = await getFunctionConfig(function_name, config);
  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }

  const inferencePromise = listInferencesWithPagination({
    function_name,
    variant_name,
    limit,
    before: beforeInference || undefined,
    after: afterInference || undefined,
  });

  const numInferencesPromise = countInferencesForVariant(
    function_name,
    variant_name,
  );
  const metricsWithFeedbackPromise = queryMetricsWithFeedback({
    function_name,
    inference_table: getInferenceTableName(function_config),
    variant_name,
  });

  const variantPerformancesPromise =
    // Only get variant performances if metric_name is provided and valid
    metric_name && config.metrics[metric_name]
      ? getVariantPerformances({
          function_name,
          function_config,
          metric_name,
          metric_config: config.metrics[metric_name],
          time_window_unit: time_granularity as TimeWindow,
          variant_name,
        })
      : undefined;

  const [
    inferenceResult,
    num_inferences,
    variant_performances,
    metricsWithFeedback,
  ] = await Promise.all([
    inferencePromise,
    numInferencesPromise,
    variantPerformancesPromise,
    metricsWithFeedbackPromise,
  ]);

  return {
    function_name,
    variant_name,
    num_inferences,
    inferences: inferenceResult.inferences,
    hasNextInferencePage: inferenceResult.hasNextPage,
    hasPreviousInferencePage: inferenceResult.hasPreviousPage,
    variant_performances,
    metricsWithFeedback,
  };
}

export default function VariantDetails({ loaderData }: Route.ComponentProps) {
  const {
    function_name,
    variant_name,
    num_inferences,
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
    variant_performances,
    metricsWithFeedback,
  } = loaderData;
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const function_config = useFunctionConfig(function_name);
  if (!function_config) {
    throw new Response(
      "Function not found. This likely means there is data in ClickHouse from an old TensorZero config.",
      {
        status: 404,
        statusText: "Not Found",
      },
    );
  }
  const variant_info = function_config.variants[variant_name];
  if (!variant_info) {
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
    searchParams.set("beforeInference", bottomInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const [metric_name, setMetricName] = useState(
    () => searchParams.get("metric_name") || "",
  );

  const handleMetricChange = (metric: string) => {
    setMetricName(metric);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("metric_name", metric);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const metricsExcludingDemonstrations = useMemo(
    () => ({
      metrics: metricsWithFeedback.metrics.filter(
        ({ metric_type }) => metric_type !== "demonstration",
      ),
    }),
    [metricsWithFeedback],
  );

  const function_type = function_config.type;
  return (
    <PageLayout>
      <PageHeader label="Variant" name={variant_name} />

      <SectionsGroup>
        <SectionLayout>
          <BasicInfo
            variantConfig={variant_info.inner}
            function_name={function_name}
            function_type={function_type}
          />
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
              singleVariantMode
            />
          )}
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Templates" />
          <VariantTemplate variantConfig={variant_info.inner} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Inferences" count={num_inferences} />
          {inferences.length > 0 ? (
            <>
              <VariantInferenceTable inferences={inferences} />
              <PageButtons
                onPreviousPage={handlePreviousInferencePage}
                onNextPage={handleNextInferencePage}
                disablePrevious={!hasPreviousInferencePage}
                disableNext={!hasNextInferencePage}
              />
            </>
          ) : (
            <div className="rounded-lg border border-gray-200 p-4 text-center text-gray-500">
              No inferences found
            </div>
          )}
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
