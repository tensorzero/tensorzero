import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow } from "~/types/tensorzero";
import type { getConfig } from "~/utils/config/index.server";

export type MetricsSectionData = {
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

export async function fetchMetricsSectionData(params: {
  function_name: string;
  metric_name: string | undefined;
  time_granularity: TimeWindow;
  config: Awaited<ReturnType<typeof getConfig>>;
}): Promise<MetricsSectionData> {
  const { function_name, metric_name, time_granularity, config } = params;

  const client = getTensorZeroClient();

  const [metricsWithFeedback, variant_performances] = await Promise.all([
    client.getFunctionMetricsWithFeedback(function_name),
    metric_name && config.metrics[metric_name]
      ? client
          .getVariantPerformances(function_name, metric_name, time_granularity)
          .then((response) =>
            response.performances.length > 0
              ? response.performances
              : undefined,
          )
      : Promise.resolve(undefined),
  ]);

  return { metricsWithFeedback, variant_performances };
}
