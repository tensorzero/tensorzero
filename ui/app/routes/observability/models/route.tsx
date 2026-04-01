import { data } from "react-router";
import type { Route } from "./+types/route";
import type { RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Models"],
};
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow, ModelUsageTimePoint } from "~/types/tensorzero";
import { ModelUsage } from "~/components/model/ModelUsage";
import { ModelLatency } from "~/components/model/ModelLatency";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { useConfig } from "~/context/config";
import { StatsBar, StatsBarSkeleton } from "~/components/ui/StatsBar";
import { Suspense } from "react";
import { Await } from "react-router";
import { formatCost } from "~/utils/cost";
import { formatCompactNumber } from "~/utils/chart";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const usageTimeGranularityParam =
    url.searchParams.get("usageTimeGranularity") || "week";
  const latencyTimeGranularityParam =
    url.searchParams.get("latencyTimeGranularity") || "week";

  // Validate TimeWindow type
  const validTimeWindows: TimeWindow[] = [
    "hour",
    "day",
    "week",
    "month",
    "cumulative",
  ];
  if (!validTimeWindows.includes(usageTimeGranularityParam as TimeWindow)) {
    throw data(
      `Invalid usage time granularity: ${usageTimeGranularityParam}. Must be one of: ${validTimeWindows.join(", ")}`,
      { status: 400 },
    );
  }
  const usageTimeGranularity = usageTimeGranularityParam as TimeWindow;
  if (!validTimeWindows.includes(latencyTimeGranularityParam as TimeWindow)) {
    throw data(
      `Invalid latency time granularity: ${latencyTimeGranularityParam}. Must be one of: ${validTimeWindows.join(", ")}`,
      { status: 400 },
    );
  }
  const latencyTimeGranularity = latencyTimeGranularityParam as TimeWindow;

  const numPeriods = parseInt(url.searchParams.get("usageNumPeriods") || "10");
  const client = getTensorZeroClient();
  const modelUsageTimeseriesPromise = client
    .getModelUsageTimeseries(usageTimeGranularity, numPeriods)
    .then((response) => response.data);
  const modelLatencyQuantilesPromise = client.getModelLatencyQuantiles(
    latencyTimeGranularity,
  );
  return {
    modelUsageTimeseriesPromise,
    usageTimeGranularity,
    latencyTimeGranularity,
    modelLatencyQuantilesPromise,
  };
}

function ModelsSummary({
  usageDataPromise,
}: {
  usageDataPromise: Promise<ModelUsageTimePoint[]>;
}) {
  const config = useConfig();
  const modelCount = config.model_names.length;

  return (
    <Suspense fallback={<StatsBarSkeleton count={5} />}>
      <Await resolve={usageDataPromise}>
        {(usageData) => {
          const filtered = usageData.filter(
            (row) => row.count && Number(row.count) > 0,
          );

          let totalInferences = 0;
          let totalInputTokens = 0;
          let totalOutputTokens = 0;
          let totalCost = 0;
          let hasCost = false;
          const activeModels = new Set<string>();

          for (const row of filtered) {
            totalInferences += Number(row.count ?? 0);
            totalInputTokens += Number(row.input_tokens ?? 0);
            totalOutputTokens += Number(row.output_tokens ?? 0);
            if (row.cost != null) {
              totalCost += row.cost;
              hasCost = true;
            }
            activeModels.add(row.model_name);
          }

          const totalTokens = totalInputTokens + totalOutputTokens;

          const items = [
            {
              label: "Models",
              value: String(modelCount),
              detail: `${activeModels.size} active`,
            },
            {
              label: "Total Inferences",
              value: formatCompactNumber(totalInferences),
            },
            {
              label: "Total Tokens",
              value: formatCompactNumber(totalTokens),
              detail: `${formatCompactNumber(totalInputTokens)} in / ${formatCompactNumber(totalOutputTokens)} out`,
            },
            ...(hasCost
              ? [{ label: "Total Cost", value: formatCost(totalCost) }]
              : []),
          ];

          return <StatsBar items={items} />;
        }}
      </Await>
    </Suspense>
  );
}

export default function ModelsPage({ loaderData }: Route.ComponentProps) {
  const { modelUsageTimeseriesPromise, modelLatencyQuantilesPromise } =
    loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Models" />
      <ModelsSummary usageDataPromise={modelUsageTimeseriesPromise} />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage" />
          <ModelUsage modelUsageDataPromise={modelUsageTimeseriesPromise} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Latency" />
          <ModelLatency
            modelLatencyResponsePromise={modelLatencyQuantilesPromise}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
