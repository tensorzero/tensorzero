import type { Route } from "./+types/route";
import { data, useLocation } from "react-router";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { useAutopilotAvailable } from "~/context/autopilot-available";
import { SnapshotBanner } from "~/components/layout/SnapshotBanner";
import { useSnapshotHash } from "~/hooks/use-snapshot-hash";
import {
  getConfigFromRequest,
  getFunctionConfig,
} from "~/utils/config/index.server";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
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
import { fetchVariantsSectionData } from "./variants-data.server";
import { VariantsSection } from "./VariantsSection";
import { fetchExperimentationSectionData } from "./experimentation-data.server";
import { ExperimentationSection } from "./ExperimentationSection";
import { fetchThroughputSectionData } from "./throughput-data.server";
import { ThroughputSection } from "./ThroughputSection";
import { fetchVariantUsageSectionData } from "./variant-usage-data.server";
import { VariantUsageSection } from "./VariantUsageSection";
import { fetchMetricsSectionData } from "./metrics-data.server";
import { MetricsSection } from "./MetricsSection";
import {
  countInferences,
  fetchInferencesSectionData,
} from "./inferences-data.server";
import { InferencesSection } from "./InferencesSection";

function FunctionDetailPageHeader({
  functionName,
  functionConfig,
}: {
  functionName: string;
  functionConfig: FunctionConfig | null;
}) {
  const autopilotAvailable = useAutopilotAvailable();
  const snapshotHash = useSnapshotHash();

  return (
    <PageHeader
      banner={snapshotHash ? <SnapshotBanner /> : undefined}
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

export async function loader({ request, params }: Route.LoaderArgs) {
  const { function_name } = params;
  const config = await getConfigFromRequest(request);
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const limit = Number(url.searchParams.get("limit")) || 10;
  const metric_name = url.searchParams.get("metric_name") || undefined;
  const time_granularity = (url.searchParams.get("time_granularity") ||
    "week") as TimeWindow;
  const throughput_time_granularity = (url.searchParams.get(
    "throughput_time_granularity",
  ) || "week") as TimeWindow;
  const variant_usage_time_granularity = (url.searchParams.get(
    "variantUsageTimeGranularity",
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

  const inferenceCountPromise = countInferences(function_name);

  return {
    function_name,
    function_config,
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
    variantUsageData: fetchVariantUsageSectionData({
      function_name,
      time_granularity: variant_usage_time_granularity,
    }),
    metricsData: fetchMetricsSectionData({
      function_name,
      metric_name,
      time_granularity,
      config,
    }),
    inferenceCountPromise,
    inferencesData: fetchInferencesSectionData({
      function_name,
      beforeInference,
      afterInference,
      limit,
    }),
  };
}

export default function FunctionDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    function_name,
    function_config,
    variantsData,
    experimentationData,
    throughputData,
    variantUsageData,
    metricsData,
    inferenceCountPromise,
    inferencesData,
  } = loaderData;
  const location = useLocation();

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

        <VariantUsageSection
          variantUsageData={variantUsageData}
          locationKey={location.key}
        />

        <MetricsSection metricsData={metricsData} locationKey={location.key} />

        <SectionLayout>
          <SectionHeader heading="Schemas" />
          <FunctionSchema functionConfig={function_config} />
        </SectionLayout>

        <InferencesSection
          inferencesData={inferencesData}
          countPromise={inferenceCountPromise}
          locationKey={location.key}
        />
      </SectionsGroup>
    </PageLayout>
  );
}
