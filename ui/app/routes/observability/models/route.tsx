import { data, useNavigate } from "react-router";
import type { Route } from "./+types/route";
import type { RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Models"],
};
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { TimeWindow } from "~/types/tensorzero";
import { getQuantiles } from "tensorzero-node";
import { ModelUsage } from "~/components/model/ModelUsage";
import { ModelLatency } from "~/components/model/ModelLatency";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { useState } from "react";

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
  const databaseClient = await getNativeDatabaseClient();
  const modelUsageTimeseriesPromise = databaseClient.getModelUsageTimeseries(
    usageTimeGranularity,
    numPeriods,
  );
  const modelLatencyQuantilesPromise = databaseClient.getModelLatencyQuantiles(
    latencyTimeGranularity,
  );
  const quantiles = getQuantiles();
  return {
    modelUsageTimeseriesPromise,
    usageTimeGranularity,
    latencyTimeGranularity,
    modelLatencyQuantilesPromise,
    quantiles,
  };
}

export default function ModelsPage({ loaderData }: Route.ComponentProps) {
  const {
    modelUsageTimeseriesPromise,
    usageTimeGranularity,
    latencyTimeGranularity,
    modelLatencyQuantilesPromise,
    quantiles,
  } = loaderData;
  const navigate = useNavigate();

  const [currentUsageTimeGranularity, setCurrentUsageTimeGranularity] =
    useState<TimeWindow>(usageTimeGranularity);
  const [currentLatencyTimeGranularity, setCurrentLatencyTimeGranularity] =
    useState<TimeWindow>(latencyTimeGranularity);

  const handleUsageTimeGranularityChange = (granularity: TimeWindow) => {
    setCurrentUsageTimeGranularity(granularity);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("usageTimeGranularity", granularity);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleLatencyTimeGranularityChange = (granularity: TimeWindow) => {
    setCurrentLatencyTimeGranularity(granularity);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("latencyTimeGranularity", granularity);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader name="Models" />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage" />
          <ModelUsage
            modelUsageDataPromise={modelUsageTimeseriesPromise}
            timeGranularity={currentUsageTimeGranularity}
            onTimeGranularityChange={handleUsageTimeGranularityChange}
          />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Latency" />
          <ModelLatency
            modelLatencyDataPromise={modelLatencyQuantilesPromise}
            quantiles={quantiles}
            timeGranularity={currentLatencyTimeGranularity}
            onTimeGranularityChange={handleLatencyTimeGranularityChange}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
