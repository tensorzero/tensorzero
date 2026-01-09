import { data, type RouteHandle } from "react-router";
import type { Route } from "./+types/route";
import { PageErrorContent } from "~/components/ui/error";
import { logger } from "~/utils/logger";
import { isInfraError } from "~/utils/tensorzero/errors";

export const handle: RouteHandle = {
  crumb: () => ["Models"],
};
import { getTensorZeroClient } from "~/utils/tensorzero.server";
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
  // Graceful degradation: return empty data on infra errors
  const modelUsageTimeseriesPromise = client
    .getModelUsageTimeseries(usageTimeGranularity, numPeriods)
    .then((response) => response.data)
    .catch((error) => {
      if (isInfraError(error)) {
        logger.warn("Infrastructure unavailable, showing degraded model usage");
        return [];
      }
      throw error;
    });
  const modelLatencyQuantilesPromise = client
    .getModelLatencyQuantiles(latencyTimeGranularity)
    .then((response) => response.data)
    .catch((error) => {
      if (isInfraError(error)) {
        logger.warn(
          "Infrastructure unavailable, showing degraded model latency",
        );
        return [];
      }
      throw error;
    });
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

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <PageErrorContent error={error} />;
}
