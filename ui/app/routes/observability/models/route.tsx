import { data } from "react-router";
import type { Route } from "./+types/route";
import type { RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Models"],
};
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow } from "~/types/tensorzero";

// Quantiles used for TDigest latency percentiles (from migration_0037)
const QUANTILES: number[] = [
  0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12,
  0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38,
  0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64,
  0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9,
  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993,
  0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
];
import { ModelUsage } from "~/components/model/ModelUsage";
import { ModelLatency } from "~/components/model/ModelLatency";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";

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
  const modelLatencyQuantilesPromise = client
    .getModelLatencyQuantiles(latencyTimeGranularity)
    .then((response) => response.data);
  return {
    modelUsageTimeseriesPromise,
    usageTimeGranularity,
    latencyTimeGranularity,
    modelLatencyQuantilesPromise,
    quantiles: QUANTILES,
  };
}

export default function ModelsPage({ loaderData }: Route.ComponentProps) {
  const {
    modelUsageTimeseriesPromise,
    modelLatencyQuantilesPromise,
    quantiles,
  } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Models" />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage" />
          <ModelUsage modelUsageDataPromise={modelUsageTimeseriesPromise} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Latency" />
          <ModelLatency
            modelLatencyDataPromise={modelLatencyQuantilesPromise}
            quantiles={quantiles}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
