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
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { useAutopilotAvailable } from "~/context/autopilot-available";
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

type MetricsData = {
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

type InferencesData = {
  inferences: Awaited<
    ReturnType<ReturnType<typeof getTensorZeroClient>["listInferenceMetadata"]>
  >["inference_metadata"];
  num_inferences: number;
  hasNextInferencePage: boolean;
  hasPreviousInferencePage: boolean;
};

async function fetchMetricsData(
  function_name: string,
  variant_name: string,
  metric_name: string | undefined,
  time_granularity: string,
  config: Awaited<ReturnType<typeof getConfig>>,
): Promise<MetricsData> {
  const client = getTensorZeroClient();

  const [metricsWithFeedback, variant_performances] = await Promise.all([
    client.getFunctionMetricsWithFeedback(function_name, variant_name),
    metric_name && config.metrics[metric_name]
      ? client
          .getVariantPerformances(
            function_name,
            metric_name,
            time_granularity as TimeWindow,
            variant_name,
          )
          .then((response) =>
            response.performances.length > 0
              ? response.performances
              : undefined,
          )
      : Promise.resolve(undefined),
  ]);

  return { metricsWithFeedback, variant_performances };
}

async function fetchInferencesData(
  function_name: string,
  variant_name: string,
  limit: number,
  beforeInference: string | null,
  afterInference: string | null,
): Promise<InferencesData> {
  const client = getTensorZeroClient();

  const [inferenceResult, num_inferences] = await Promise.all([
    client.listInferenceMetadata({
      function_name,
      variant_name,
      limit: limit + 1, // Fetch one extra to determine pagination
      before: beforeInference || undefined,
      after: afterInference || undefined,
    }),
    countInferencesForVariant(function_name, variant_name),
  ]);

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
    num_inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
  };
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

  return {
    function_name,
    variant_name,
    metricsData: fetchMetricsData(
      function_name,
      variant_name,
      metric_name,
      time_granularity,
      config,
    ),
    inferencesData: fetchInferencesData(
      function_name,
      variant_name,
      limit,
      beforeInference,
      afterInference,
    ),
  };
}

function MetricsSkeleton() {
  return (
    <>
      <Skeleton className="mb-4 h-10 w-64" />
      <Skeleton className="h-64 w-full" />
    </>
  );
}

function MetricsSectionError() {
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
      title="Error loading metrics"
      description={message}
    />
  );
}

function InferencesSkeleton() {
  return (
    <>
      <Skeleton className="mb-2 h-6 w-32" />
      <Skeleton className="h-64 w-full" />
    </>
  );
}

function InferencesSectionError() {
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

function MetricsContent({ data }: { data: MetricsData }) {
  const { metricsWithFeedback, variant_performances } = data;
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
    </>
  );
}

function InferencesContent({ data }: { data: InferencesData }) {
  const {
    inferences,
    num_inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
  } = data;
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

  return (
    <>
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
    </>
  );
}

function VariantDetailPageHeader({
  functionName,
  variantName,
}: {
  functionName: string;
  variantName: string;
}) {
  const autopilotAvailable = useAutopilotAvailable();

  return (
    <PageHeader
      eyebrow={
        <Breadcrumbs
          segments={[
            { label: "Functions", href: "/observability/functions" },
            {
              label: functionName,
              href: toFunctionUrl(functionName),
              isIdentifier: true,
            },
            { label: "Variants" },
          ]}
        />
      }
      name={variantName}
    >
      {autopilotAvailable && (
        <AskAutopilotButton
          message={`Variant: ${variantName}\nFunction: ${functionName}\n\n`}
        />
      )}
    </PageHeader>
  );
}

export default function VariantDetails({ loaderData }: Route.ComponentProps) {
  const { function_name, variant_name, metricsData, inferencesData } =
    loaderData;
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
      <VariantDetailPageHeader
        functionName={function_name}
        variantName={variant_name}
      />

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
          <Suspense
            key={`metrics-${location.key}`}
            fallback={<MetricsSkeleton />}
          >
            <Await resolve={metricsData} errorElement={<MetricsSectionError />}>
              {(data) => <MetricsContent data={data} />}
            </Await>
          </Suspense>
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Templates" />
          <VariantTemplate variantConfig={variant_info.inner} />
        </SectionLayout>

        <SectionLayout>
          <Suspense
            key={`inferences-${location.key}`}
            fallback={<InferencesSkeleton />}
          >
            <Await
              resolve={inferencesData}
              errorElement={<InferencesSectionError />}
            >
              {(data) => <InferencesContent data={data} />}
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
