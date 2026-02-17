import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import {
  Await,
  data,
  useAsyncError,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { useAutopilotAvailable } from "~/context/autopilot-available";
import PageButtons from "~/components/utils/PageButtons";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import FunctionInferenceTable from "./FunctionInferenceTable";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
import { useFunctionConfig } from "~/context/config";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { Suspense, useMemo } from "react";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
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
import type { FunctionConfig, TimeWindow } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { applyPaginationLogic } from "~/utils/pagination";
import { Skeleton } from "~/components/ui/skeleton";
import { PageErrorContent } from "~/components/ui/error";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { fetchVariantsSectionData } from "./variants-data.server";
import { VariantsSection } from "./VariantsSection";
import { fetchExperimentationSectionData } from "./experimentation-data.server";
import { ExperimentationSection } from "./ExperimentationSection";
import { fetchThroughputSectionData } from "./throughput-data.server";
import { ThroughputSection } from "./ThroughputSection";

export type FunctionDetailData = Awaited<
  ReturnType<typeof fetchFunctionDetailData>
>;

function FunctionDetailPageHeader({
  functionName,
  functionConfig,
}: {
  functionName: string;
  functionConfig: FunctionConfig | null;
}) {
  const autopilotAvailable = useAutopilotAvailable();

  return (
    <PageHeader
      eyebrow={
        <Breadcrumbs
          segments={[{ label: "Functions", href: "/observability/functions" }]}
        />
      }
      name={functionName}
      tag={
        functionConfig ? (
          <FunctionTypeBadge type={functionConfig.type} />
        ) : undefined
      }
    >
      {functionConfig && <BasicInfo functionConfig={functionConfig} />}
      {autopilotAvailable && (
        <AskAutopilotButton message={`Function: ${functionName}\n\n`} />
      )}
    </PageHeader>
  );
}

function SectionsSkeleton() {
  return (
    <>
      <SectionLayout>
        <SectionHeader heading="Metrics" />
        <Skeleton className="mb-4 h-10 w-64" />
        <Skeleton className="h-64 w-full" />
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Schemas" />
        <Skeleton className="h-32 w-full" />
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Inferences" />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Variant</TableHead>
              <TableHead>Time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3, 4, 5].map((i) => (
              <TableRow key={i}>
                <TableCell>
                  <Skeleton className="h-4 w-48" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-32" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionLayout>
    </>
  );
}

function SectionsErrorState() {
  const error = useAsyncError();
  return (
    <SectionLayout>
      <PageErrorContent error={error} />
    </SectionLayout>
  );
}

type FetchParams = {
  function_name: string;
  config: Awaited<ReturnType<typeof getConfig>>;
  beforeInference: string | null;
  afterInference: string | null;
  limit: number;
  metric_name: string | undefined;
  time_granularity: TimeWindow;
};

async function fetchFunctionDetailData(params: FetchParams) {
  const {
    function_name,
    config,
    beforeInference,
    afterInference,
    limit,
    metric_name,
    time_granularity,
  } = params;

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
  const [
    inferenceResult,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
  ] = await Promise.all([
    inferencePromise,
    numInferencesPromise,
    metricsWithFeedbackPromise,
    variantPerformancesPromise,
  ]);

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
  };
}

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

  return {
    function_name,
    variantsData: fetchVariantsSectionData({ function_name, function_config }),
    experimentationData:
      function_name !== DEFAULT_FUNCTION
        ? fetchExperimentationSectionData({
            function_name,
            function_config,
            time_granularity: feedback_time_granularity,
          })
        : null,
    throughputData: fetchThroughputSectionData({
      function_name,
      time_granularity: throughput_time_granularity,
    }),
    functionDetailData: fetchFunctionDetailData({
      function_name,
      config,
      beforeInference,
      afterInference,
      limit,
      metric_name,
      time_granularity,
    }),
  };
}

function SectionsContent({
  data,
  functionConfig,
}: {
  data: FunctionDetailData;
  functionConfig: FunctionConfig;
}) {
  const {
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
    num_inferences,
    metricsWithFeedback,
    variant_performances,
  } = data;

  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

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
    <>
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
        <FunctionSchema functionConfig={functionConfig} />
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
    </>
  );
}

export default function FunctionDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    function_name,
    variantsData,
    experimentationData,
    throughputData,
    functionDetailData,
  } = loaderData;
  const location = useLocation();
  const function_config = useFunctionConfig(function_name);

  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }

  return (
    <PageLayout>
      <FunctionDetailPageHeader
        functionName={function_name}
        functionConfig={function_config}
      />

      <SectionsGroup>
        <VariantsSection
          variantsData={variantsData}
          functionName={function_name}
          locationKey={location.key}
        />

        {experimentationData && (
          <ExperimentationSection
            experimentationData={experimentationData}
            functionConfig={function_config}
            functionName={function_name}
            locationKey={location.key}
          />
        )}

        <ThroughputSection
          throughputData={throughputData}
          locationKey={location.key}
        />

        <Suspense key={location.key} fallback={<SectionsSkeleton />}>
          <Await
            resolve={functionDetailData}
            errorElement={<SectionsErrorState />}
          >
            {(data) => (
              <SectionsContent data={data} functionConfig={function_config} />
            )}
          </Await>
        </Suspense>
      </SectionsGroup>
    </PageLayout>
  );
}
