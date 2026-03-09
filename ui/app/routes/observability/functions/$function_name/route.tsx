import type { Route } from "./+types/route";
import { data, useLocation, useSearchParams } from "react-router";
import { useCallback, useMemo } from "react";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { useAutopilotAvailable } from "~/context/autopilot-available";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import BasicInfo from "./FunctionBasicInfo";
import FunctionSchema from "./FunctionSchema";
import { useFunctionConfig } from "~/context/config";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { FunctionTypeBadge } from "~/components/function/FunctionSelector";
import { NamespaceSelector } from "~/components/function/NamespaceSelector";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type { FunctionConfig, TimeWindow } from "~/types/tensorzero";
import { fetchVariantsSectionData } from "./variants-data.server";
import { VariantsSection } from "./VariantsSection";
import { fetchExperimentationSectionData } from "./experimentation-data.server";
import { ExperimentationSection } from "./ExperimentationSection";
import { fetchThroughputSectionData } from "./throughput-data.server";
import { ThroughputSection } from "./ThroughputSection";
import { fetchMetricsSectionData } from "./metrics-data.server";
import { MetricsSection } from "./MetricsSection";
import {
  countInferences,
  fetchInferencesSectionData,
} from "./inferences-data.server";
import { InferencesSection } from "./InferencesSection";

function getNamespaceNames(functionConfig: FunctionConfig | null): string[] {
  if (!functionConfig) return [];
  return Object.keys(functionConfig.experimentation.namespaces).sort();
}

interface FunctionDetailPageHeaderProps {
  functionName: string;
  functionConfig: FunctionConfig | null;
  namespace: string | undefined;
  onNamespaceChange: (namespace: string | undefined) => void;
}

function FunctionDetailPageHeader({
  functionName,
  functionConfig,
  namespace,
  onNamespaceChange,
}: FunctionDetailPageHeaderProps) {
  const autopilotAvailable = useAutopilotAvailable();
  const namespaces = useMemo(
    () => getNamespaceNames(functionConfig),
    [functionConfig],
  );

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
      <div className="flex items-center gap-3">
        {namespaces.length > 0 && (
          <NamespaceSelector
            namespaces={namespaces}
            value={namespace}
            onChange={onNamespaceChange}
          />
        )}
        {autopilotAvailable && (
          <AskAutopilotButton message={`Function: ${functionName}\n\n`} />
        )}
      </div>
    </PageHeader>
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
  const namespace = url.searchParams.get("namespace") || undefined;
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

  const inferenceCountPromise = countInferences(function_name, namespace);

  return {
    function_name,
    namespace,
    variantsData: fetchVariantsSectionData({
      function_name,
      function_config,
      namespace,
    }),
    experimentationData:
      function_name !== DEFAULT_FUNCTION
        ? fetchExperimentationSectionData({
            function_name,
            function_config,
            time_granularity: feedback_time_granularity,
            namespace,
          })
        : null,
    throughputData: fetchThroughputSectionData({
      function_name,
      time_granularity: throughput_time_granularity,
      namespace,
    }),
    metricsData: fetchMetricsSectionData({
      function_name,
      metric_name,
      time_granularity,
      config,
      namespace,
    }),
    inferenceCountPromise,
    inferencesData: fetchInferencesSectionData({
      function_name,
      beforeInference,
      afterInference,
      limit,
      namespace,
    }),
  };
}

export default function FunctionDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    function_name,
    namespace,
    variantsData,
    experimentationData,
    throughputData,
    metricsData,
    inferenceCountPromise,
    inferencesData,
  } = loaderData;
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const function_config = useFunctionConfig(function_name);

  const handleNamespaceChange = useCallback(
    (ns: string | undefined) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (ns) {
            next.set("namespace", ns);
          } else {
            next.delete("namespace");
          }
          return next;
        },
        { preventScrollReset: true },
      );
    },
    [setSearchParams],
  );

  if (!function_config) {
    throw data(`Function ${function_name} not found`, { status: 404 });
  }

  return (
    <PageLayout>
      <FunctionDetailPageHeader
        functionName={function_name}
        functionConfig={function_config}
        namespace={namespace}
        onNamespaceChange={handleNamespaceChange}
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
            namespace={namespace}
            locationKey={location.key}
          />
        )}

        <ThroughputSection
          throughputData={throughputData}
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
