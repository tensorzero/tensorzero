import { data, useNavigate } from "react-router";
import type { Route } from "./+types/route";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { TimeWindow } from "tensorzero-node";
import { ModelUsage } from "~/components/model/ModelUsage";
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
  const timeGranularityParam =
    url.searchParams.get("usageTimeGranularity") || "week";

  // Validate TimeWindow type
  const validTimeWindows: TimeWindow[] = [
    "hour",
    "day",
    "week",
    "month",
    "cumulative",
  ];
  if (!validTimeWindows.includes(timeGranularityParam as TimeWindow)) {
    throw data(
      `Invalid time granularity: ${timeGranularityParam}. Must be one of: ${validTimeWindows.join(", ")}`,
      { status: 400 },
    );
  }
  const time_granularity = timeGranularityParam as TimeWindow;

  const numPeriods = parseInt(url.searchParams.get("usageNumPeriods") || "10");
  const databaseClient = await getNativeDatabaseClient();
  const modelUsageTimeseriesPromise = databaseClient.getModelUsageTimeseries(
    time_granularity,
    numPeriods,
  );
  return {
    modelUsageTimeseriesPromise,
    timeGranularity: time_granularity,
  };
}

export default function ModelsPage({ loaderData }: Route.ComponentProps) {
  const { modelUsageTimeseriesPromise, timeGranularity } = loaderData;
  const navigate = useNavigate();

  const [currentTimeGranularity, setCurrentTimeGranularity] =
    useState<TimeWindow>(timeGranularity);

  const handleTimeGranularityChange = (granularity: TimeWindow) => {
    setCurrentTimeGranularity(granularity);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("usageTimeGranularity", granularity);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader name="Models" label="Model Usage" />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage" />
          <ModelUsage
            modelUsageDataPromise={modelUsageTimeseriesPromise}
            timeGranularity={currentTimeGranularity}
            onTimeGranularityChange={handleTimeGranularityChange}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
