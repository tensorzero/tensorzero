import { data } from "react-router";
import type { Route } from "./+types/route";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { TimeWindow } from "tensorzero-node";

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
  const modelUsageTimeseriesPromise =
    await databaseClient.getModelUsageTimeseries(time_granularity, numPeriods);
  return {
    modelUsageTimeseriesPromise,
  };
}
