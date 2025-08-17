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
  Await,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import FunctionInferenceTable from "./FunctionInferenceTable";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
import { useFunctionConfig } from "~/context/config";
import {
  getVariantCounts,
  getVariantPerformances,
  type TimeWindowUnit,
} from "~/utils/clickhouse/function";
import { queryMetricsWithFeedback } from "~/utils/clickhouse/feedback";
import { getInferenceTableName } from "~/utils/clickhouse/common";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { useState, Suspense } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
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
          time_window_unit: time_granularity as TimeWindowUnit,
        })
      : undefined;

  const variantCountsWithMetadataPromise = variantCountsPromise.then(
    (variant_counts) =>
      variant_counts.map((variant_count) => {
        const variant_config = function_config.variants[
          variant_count.variant_name
        ] || {
          inner: {
            // In case the variant is not found, we still want to display the variant name
            type: "unknown",
            weight: 0,
          },
        };
        return {
          ...variant_count,
          type: variant_config.inner.type,
          weight: variant_config.inner.weight,
        };
      }),
  );
  return {
    function_name,
    inferencePromise,
    tableBoundsPromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
    variantCountsWithMetadataPromise,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    function_name,
    inferencePromise,
    tableBoundsPromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
    variantCountsWithMetadataPromise,
  } = loaderData;
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const function_config = useFunctionConfig(function_name);
  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }

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
          <Suspense fallback={<div>Loading variants...</div>}>
            <Await resolve={variantCountsWithMetadataPromise}>
              {(variant_counts) => (
                <FunctionVariantTable
                  variant_counts={variant_counts}
                  function_name={function_name}
                />
              )}
            </Await>
          </Suspense>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Metrics" />
          <Suspense fallback={<div>Loading metrics...</div>}>
            <Await resolve={metricsWithFeedbackPromise}>
              {(metricsWithFeedback) => {
                const metricsExcludingDemonstrations = {
                  metrics: metricsWithFeedback.metrics.filter(
                    ({ metric_type }) => metric_type !== "demonstration",
                  ),
                };

                return (
                  <>
                    <MetricSelector
                      metricsWithFeedback={metricsExcludingDemonstrations}
                      selectedMetric={metric_name || ""}
                      onMetricChange={handleMetricChange}
                    />
                    {variantPerformancesPromise && (
                      <Suspense
                        fallback={<div>Loading performance data...</div>}
                      >
                        <Await resolve={variantPerformancesPromise}>
                          {(variant_performances) =>
                            variant_performances && (
                              <VariantPerformance
                                variant_performances={variant_performances}
                                metric_name={metric_name}
                                time_granularity={time_granularity}
                                onTimeGranularityChange={
                                  handleTimeGranularityChange
                                }
                              />
                            )
                          }
                        </Await>
                      </Suspense>
                    )}
                  </>
                );
              }}
            </Await>
          </Suspense>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Schemas" />
          <FunctionSchema functionConfig={function_config} />
        </SectionLayout>

        <SectionLayout>
          <Suspense fallback={<div>Loading inference count...</div>}>
            <Await resolve={numInferencesPromise}>
              {(num_inferences) => (
                <SectionHeader heading="Inferences" count={num_inferences} />
              )}
            </Await>
          </Suspense>
          <Suspense fallback={<div>Loading inferences...</div>}>
            <Await resolve={inferencePromise}>
              {(inferences) => (
                <>
                  <FunctionInferenceTable inferences={inferences} />
                  <Suspense fallback={<div>Loading pagination...</div>}>
                    <Await resolve={tableBoundsPromise}>
                      {(inference_bounds) => {
                        const topInference =
                          inferences.length > 0 ? inferences[0] : null;
                        const bottomInference =
                          inferences.length > 0
                            ? inferences[inferences.length - 1]
                            : null;

                        const handleNextInferencePage = () => {
                          if (!bottomInference) return;
                          const searchParams = new URLSearchParams(
                            window.location.search,
                          );
                          searchParams.delete("afterInference");
                          searchParams.set(
                            "beforeInference",
                            bottomInference.id,
                          );
                          navigate(`?${searchParams.toString()}`, {
                            preventScrollReset: true,
                          });
                        };

                        const handlePreviousInferencePage = () => {
                          if (!topInference) return;
                          const searchParams = new URLSearchParams(
                            window.location.search,
                          );
                          searchParams.delete("beforeInference");
                          searchParams.set("afterInference", topInference.id);
                          navigate(`?${searchParams.toString()}`, {
                            preventScrollReset: true,
                          });
                        };

                        const disablePreviousInferencePage =
                          !topInference ||
                          inference_bounds.last_id === topInference.id;
                        const disableNextInferencePage =
                          !bottomInference ||
                          inference_bounds.first_id === bottomInference.id;

                        return (
                          <PageButtons
                            onPreviousPage={handlePreviousInferencePage}
                            onNextPage={handleNextInferencePage}
                            disablePrevious={disablePreviousInferencePage}
                            disableNext={disableNextInferencePage}
                          />
                        );
                      }}
                    </Await>
                  </Suspense>
                </>
              )}
            </Await>
          </Suspense>
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
