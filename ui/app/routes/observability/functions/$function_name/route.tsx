import type { Route } from "./+types/route";
import {
  Await,
  data,
  useAsyncError,
  useLocation,
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
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import { Suspense, useMemo } from "react";
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
import type { FunctionConfig, TimeWindow } from "~/types/tensorzero";
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
import {
  fetchAllFunctionDetailData,
  type FunctionDetailData,
} from "./function-data.server";

function FunctionDetailPageHeader({
  functionName,
  functionConfig,
}: {
  functionName: string;
  functionConfig: FunctionConfig | null;
}) {
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
    </PageHeader>
  );
}

function SectionsSkeleton() {
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Variants" />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Weight</TableHead>
              <TableHead>Inferences</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[1, 2, 3].map((i) => (
              <TableRow key={i}>
                <TableCell>
                  <Skeleton className="h-4 w-32" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-24" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-12" />
                </TableCell>
                <TableCell>
                  <Skeleton className="h-4 w-16" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Experimentation" />
        <Skeleton className="h-32 w-full" />
      </SectionLayout>

      <SectionLayout>
        <SectionHeader heading="Throughput" />
        <Skeleton className="h-64 w-full" />
      </SectionLayout>

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
    </SectionsGroup>
  );
}

function SectionsErrorState() {
  const error = useAsyncError();
  return (
    <SectionsGroup>
      <SectionLayout>
        <PageErrorContent error={error} />
      </SectionLayout>
    </SectionsGroup>
  );
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
    functionDetailData: fetchAllFunctionDetailData({
      function_name,
      function_config,
      config,
      beforeInference,
      afterInference,
      limit,
      metric_name,
      time_granularity,
      throughput_time_granularity,
      feedback_time_granularity,
    }),
  };
}

function SectionsContent({
  data,
  functionName,
  functionConfig,
}: {
  data: FunctionDetailData;
  functionName: string;
  functionConfig: FunctionConfig;
}) {
  const {
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
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Variants" />
        <FunctionVariantTable
          variant_counts={variant_counts}
          function_name={functionName}
        />
      </SectionLayout>

      {functionName !== DEFAULT_FUNCTION && (
        <SectionLayout>
          <SectionHeader heading="Experimentation" />
          <FunctionExperimentation
            functionConfig={functionConfig}
            functionName={functionName}
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
    </SectionsGroup>
  );
}

export default function FunctionDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const { function_name, functionDetailData } = loaderData;
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

      <Suspense key={location.key} fallback={<SectionsSkeleton />}>
        <Await
          resolve={functionDetailData}
          errorElement={<SectionsErrorState />}
        >
          {(data) => (
            <SectionsContent
              data={data}
              functionName={function_name}
              functionConfig={function_config}
            />
          )}
        </Await>
      </Suspense>
    </PageLayout>
  );
}
