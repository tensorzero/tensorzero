import {
  Await,
  data,
  isRouteErrorResponse,
  redirect,
  useAsyncError,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router";
import type { LoaderFunctionArgs, RouteHandle } from "react-router";
import { AlertCircle } from "lucide-react";
import { Suspense, useMemo, useState } from "react";
import BasicInfo from "./VariantBasicInfo";
import VariantTemplate from "./VariantTemplate";
import { useFunctionConfig } from "~/context/config";
import PageButtons from "~/components/utils/PageButtons";
import VariantInferenceTable from "./VariantInferenceTable";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { countInferencesForVariant } from "~/utils/clickhouse/inference.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow } from "~/types/tensorzero";
import { VariantPerformance } from "~/components/function/variant/VariantPerformance";
import { MetricSelector } from "~/components/function/variant/MetricSelector";
import type { Route } from "./+types/route";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { toFunctionUrl } from "~/utils/urls";
import { applyPaginationLogic } from "~/utils/pagination";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Variants",
    { label: match.params.variant_name!, isIdentifier: true },
  ],
};

// Type definitions for granular data fetching
type MetricsWithFeedbackData = Awaited<
  ReturnType<
    ReturnType<typeof getTensorZeroClient>["getFunctionMetricsWithFeedback"]
  >
>;

type VariantPerformanceData =
  | Awaited<
      ReturnType<
        ReturnType<typeof getTensorZeroClient>["getVariantPerformances"]
      >
    >["performances"]
  | undefined;

type InferencesTableData = {
  inferences: Awaited<
    ReturnType<ReturnType<typeof getTensorZeroClient>["listInferenceMetadata"]>
  >["inference_metadata"];
  hasNextInferencePage: boolean;
  hasPreviousInferencePage: boolean;
};

// Granular data fetching functions
async function fetchMetricsWithFeedback(
  function_name: string,
  variant_name: string,
): Promise<MetricsWithFeedbackData> {
  const client = getTensorZeroClient();
  return client.getFunctionMetricsWithFeedback(function_name, variant_name);
}

async function fetchVariantPerformance(
  function_name: string,
  variant_name: string,
  metric_name: string | undefined,
  time_granularity: string,
  config: Awaited<ReturnType<typeof getConfig>>,
): Promise<VariantPerformanceData> {
  if (!metric_name || !config.metrics[metric_name]) {
    return undefined;
  }
  const client = getTensorZeroClient();
  const response = await client.getVariantPerformances(
    function_name,
    metric_name,
    time_granularity as TimeWindow,
    variant_name,
  );
  return response.performances.length > 0 ? response.performances : undefined;
}

async function fetchInferencesTable(
  function_name: string,
  variant_name: string,
  limit: number,
  beforeInference: string | null,
  afterInference: string | null,
): Promise<InferencesTableData> {
  const client = getTensorZeroClient();
  const inferenceResult = await client.listInferenceMetadata({
    function_name,
    variant_name,
    limit: limit + 1, // Fetch one extra to determine pagination
    before: beforeInference || undefined,
    after: afterInference || undefined,
  });

  const {
    items: inferences,
    hasNextPage: hasNextInferencePage,
    hasPreviousPage: hasPreviousInferencePage,
  } = applyPaginationLogic(inferenceResult.inference_metadata, limit, {
    before: beforeInference,
    after: afterInference,
  });

  return {
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
  };
}

async function fetchInferenceCount(
  function_name: string,
  variant_name: string,
): Promise<number> {
  return countInferencesForVariant(function_name, variant_name);
}

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

  // Return granular promises for independent streaming
  return {
    function_name,
    variant_name,
    // Metrics section - split into selector data and chart data
    metricsWithFeedbackData: fetchMetricsWithFeedback(
      function_name,
      variant_name,
    ),
    variantPerformanceData: fetchVariantPerformance(
      function_name,
      variant_name,
      metric_name,
      time_granularity,
      config,
    ),
    // Inferences section - split into table data and count
    inferencesTableData: fetchInferencesTable(
      function_name,
      variant_name,
      limit,
      beforeInference,
      afterInference,
    ),
    inferenceCountData: fetchInferenceCount(function_name, variant_name),
  };
}

// Skeleton components
function MetricSelectorSkeleton() {
  return <Skeleton className="h-10 w-64" />;
}

function VariantPerformanceSkeleton() {
  return <Skeleton className="mt-4 h-64 w-full" />;
}

function InferencesTableSkeleton() {
  return <Skeleton className="h-64 w-full" />;
}

// Error components
function MetricSelectorError() {
  const error = useAsyncError();
  let message = "Failed to load metrics";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading metric selector"
      description={message}
    />
  );
}

function VariantPerformanceError() {
  const error = useAsyncError();
  let message = "Failed to load performance data";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading performance chart"
      description={message}
    />
  );
}

function InferencesTableError() {
  const error = useAsyncError();
  let message = "Failed to load inferences";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading inferences"
      description={message}
    />
  );
}

// Content components
function MetricSelectorContent({
  data,
  onMetricChange,
  selectedMetric,
}: {
  data: MetricsWithFeedbackData;
  onMetricChange: (metric: string) => void;
  selectedMetric: string;
}) {
  const metricsExcludingDemonstrations = useMemo(
    () => ({
      metrics: data.metrics.filter(
        ({ metric_type }) => metric_type !== "demonstration",
      ),
    }),
    [data],
  );

  return (
    <MetricSelector
      metricsWithFeedback={metricsExcludingDemonstrations}
      selectedMetric={selectedMetric}
      onMetricChange={onMetricChange}
    />
  );
}

function VariantPerformanceContent({
  data,
  metric_name,
}: {
  data: VariantPerformanceData;
  metric_name: string;
}) {
  if (!data) {
    return null;
  }
  return (
    <VariantPerformance
      variant_performances={data}
      metric_name={metric_name}
      singleVariantMode
    />
  );
}

function InferencesTableContent({ data }: { data: InferencesTableData }) {
  const { inferences, hasNextInferencePage, hasPreviousInferencePage } = data;
  const navigate = useNavigate();

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

  if (inferences.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 p-4 text-center text-gray-500">
        No inferences found
      </div>
    );
  }

  return (
    <>
      <VariantInferenceTable inferences={inferences} />
      <PageButtons
        onPreviousPage={handlePreviousInferencePage}
        onNextPage={handleNextInferencePage}
        disablePrevious={!hasPreviousInferencePage}
        disableNext={!hasNextInferencePage}
      />
    </>
  );
}

// Section components with Suspense boundaries
function MetricsSection({
  metricsWithFeedbackData,
  variantPerformanceData,
  locationKey,
}: {
  metricsWithFeedbackData: Promise<MetricsWithFeedbackData>;
  variantPerformanceData: Promise<VariantPerformanceData>;
  locationKey: string | undefined;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [metric_name, setMetricName] = useState(
    () => searchParams.get("metric_name") || "",
  );

  const handleMetricChange = (metric: string) => {
    setMetricName(metric);
    const newSearchParams = new URLSearchParams(window.location.search);
    newSearchParams.set("metric_name", metric);
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <SectionLayout>
      <SectionHeader heading="Metrics" />
      <Suspense
        key={`metric-selector-${locationKey}`}
        fallback={<MetricSelectorSkeleton />}
      >
        <Await
          resolve={metricsWithFeedbackData}
          errorElement={<MetricSelectorError />}
        >
          {(data) => (
            <MetricSelectorContent
              data={data}
              onMetricChange={handleMetricChange}
              selectedMetric={metric_name}
            />
          )}
        </Await>
      </Suspense>
      <Suspense
        key={`variant-performance-${locationKey}`}
        fallback={<VariantPerformanceSkeleton />}
      >
        <Await
          resolve={variantPerformanceData}
          errorElement={<VariantPerformanceError />}
        >
          {(data) => (
            <VariantPerformanceContent data={data} metric_name={metric_name} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function InferencesSection({
  inferencesTableData,
  inferenceCountData,
  locationKey,
}: {
  inferencesTableData: Promise<InferencesTableData>;
  inferenceCountData: Promise<number>;
  locationKey: string | undefined;
}) {
  return (
    <SectionLayout>
      <SectionHeader heading="Inferences" count={inferenceCountData} />
      <Suspense
        key={`inferences-table-${locationKey}`}
        fallback={<InferencesTableSkeleton />}
      >
        <Await
          resolve={inferencesTableData}
          errorElement={<InferencesTableError />}
        >
          {(data) => <InferencesTableContent data={data} />}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

export default function VariantDetails({ loaderData }: Route.ComponentProps) {
  const {
    function_name,
    variant_name,
    metricsWithFeedbackData,
    variantPerformanceData,
    inferencesTableData,
    inferenceCountData,
  } = loaderData;
  const location = useLocation();
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

  let variant_info = function_config.variants[variant_name];
  if (!variant_info) {
    if (function_name === DEFAULT_FUNCTION) {
      // For default function, create synthetic variant config
      variant_info = {
        inner: {
          type: "chat_completion",
          model: variant_name,
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
    } else {
      throw new Response(
        "Variant not found. This likely means there is data in ClickHouse from an old TensorZero config.",
        {
          status: 404,
          statusText: "Not Found",
        },
      );
    }
  }

  const function_type = function_config.type;

  return (
    <PageLayout>
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[
              { label: "Functions", href: "/observability/functions" },
              {
                label: function_name,
                href: toFunctionUrl(function_name),
                isIdentifier: true,
              },
              { label: "Variants" },
            ]}
          />
        }
        name={variant_name}
      />

      <SectionsGroup>
        <SectionLayout>
          <BasicInfo
            variantConfig={variant_info.inner}
            function_name={function_name}
            function_type={function_type}
          />
        </SectionLayout>

        <MetricsSection
          metricsWithFeedbackData={metricsWithFeedbackData}
          variantPerformanceData={variantPerformanceData}
          locationKey={location.key}
        />

        <SectionLayout>
          <SectionHeader heading="Templates" />
          <VariantTemplate variantConfig={variant_info.inner} />
        </SectionLayout>

        <InferencesSection
          inferencesTableData={inferencesTableData}
          inferenceCountData={inferenceCountData}
          locationKey={location.key}
        />
      </SectionsGroup>
    </PageLayout>
  );
}
