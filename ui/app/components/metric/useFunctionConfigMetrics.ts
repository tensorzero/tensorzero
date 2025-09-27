import { useEffect, useMemo } from "react";
import {
  useWatch,
  type Control,
  type FieldPathValue,
  type Path,
} from "react-hook-form";
import { useFetcher, type FetcherWithComponents } from "react-router";
import type { Config } from "tensorzero-node";
import type { MetricsWithFeedbackData } from "~/utils/clickhouse/feedback";
import type { FeedbackConfig } from "~/utils/config/feedback";

export interface UseFunctionConfigMetricsProps<
  T extends Record<string, unknown>,
> {
  control: Control<T>;
  functionFieldName: Path<T>;
  config: Config;
  addDemonstrations: boolean;
  // Notifies caller when this hooks's internal fetcher is loading
  onMetricsLoadingChange?: (loading: boolean) => void;
}

export interface UseFunctionConfigMetricsReturn<
  T extends Record<string, unknown>,
> {
  metricsFetcher: FetcherWithComponents<MetricsWithFeedbackData>;
  metrics: Record<string, FeedbackConfig>;
  functionValue: FieldPathValue<T, Path<T>>;
  metricsLoading: boolean;
  validMetrics: Set<string>;
}

export function useFunctionConfigMetrics<T extends Record<string, unknown>>({
  control,
  functionFieldName,
  config,
  addDemonstrations,
  onMetricsLoadingChange,
}: UseFunctionConfigMetricsProps<T>): UseFunctionConfigMetricsReturn<T> {
  const metricsFetcher = useFetcher<MetricsWithFeedbackData>();

  const metrics = Object.fromEntries(
    Object.entries(config.metrics).filter(([, v]) => v !== undefined),
  ) as Record<string, FeedbackConfig>;

  if (addDemonstrations) {
    metrics["demonstration"] = {
      type: "demonstration",
      level: "inference",
    };
  }

  const functionValue = useWatch({
    control,
    name: functionFieldName,
  });

  useEffect(() => {
    if (functionValue && typeof functionValue === "string") {
      metricsFetcher.load(
        `/api/function/${encodeURIComponent(functionValue)}/feedback_counts`,
      );
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [functionValue]);

  const validMetrics = useMemo(() => {
    if (!metricsFetcher.data) return new Set<string>();
    return new Set(
      metricsFetcher.data.metrics
        .filter((m) => addDemonstrations || m.metric_name !== "demonstration")
        .map((m) => m.metric_name),
    );
  }, [metricsFetcher.data, addDemonstrations]);

  const metricsLoading = metricsFetcher.state === "loading";

  // Inform parent when the internal metrics fetcher loading state changes
  useEffect(() => {
    onMetricsLoadingChange?.(metricsLoading);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metricsLoading]);

  return {
    metricsFetcher,
    metrics,
    functionValue,
    metricsLoading,
    validMetrics,
  };
}
