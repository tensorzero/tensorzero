import type {
  Config,
  MetricConfig,
  MetricConfigLevel,
  MetricConfigOptimize,
} from "tensorzero-node";
import { z } from "zod";

export const FeedbackTypeSchema = z.enum([
  "boolean",
  "float",
  "comment",
  "demonstration",
]);
export type FeedbackType = z.infer<typeof FeedbackTypeSchema>;

export type FeedbackConfig =
  | MetricConfig
  | { type: "comment" }
  | { type: "demonstration"; level: "inference" };

export function getFeedbackConfig(
  metricName: string,
  config: Config,
): FeedbackConfig | undefined {
  if (metricName === "comment") {
    return { type: "comment" };
  } else if (metricName === "demonstration") {
    return { type: "demonstration", level: "inference" };
  }
  const metric = config.metrics[metricName];
  if (!metric) {
    return undefined;
  }
  return metric;
}

/**
 * Returns the appropriate comparison operator based on the optimization direction
 * @param optimize The optimization direction ("min" or "max")
 * @returns The comparison operator (">" or "<")
 */
export function getComparisonOperator(
  optimize: MetricConfigOptimize,
): "<" | ">" {
  return optimize === "max" ? ">" : "<";
}

export function filterMetricsByLevel(
  metrics: { [x: string]: MetricConfig | undefined },
  level: MetricConfigLevel,
): { [x: string]: FeedbackConfig | undefined } {
  // First filter the metrics by level
  const filteredEntries: [string, FeedbackConfig][] = Object.entries(metrics)
    .filter(([, metric]) => metric?.level === level)
    .filter(([, metric]) => metric !== undefined) as [string, FeedbackConfig][];

  // Prepare comment and demonstration entries
  const specialEntries: [string, FeedbackConfig][] = [];

  // Add comment to specialEntries unconditionally
  specialEntries.push(["comment", { type: "comment" }]);

  // Add demonstration to specialEntries if level is inference
  if (level === "inference") {
    specialEntries.push([
      "demonstration",
      { type: "demonstration", level: "inference" },
    ]);
  }

  // Return object with comment and demonstration first, then the rest
  return Object.fromEntries([...specialEntries, ...filteredEntries]);
}

// Removes metrics that are part of a static evaluation
// These will have names that start with "tensorzero::evaluation_name::"
export function filterStaticEvaluationMetrics(metrics: {
  [x: string]: FeedbackConfig | undefined;
}) {
  return Object.fromEntries(
    Object.entries(metrics).filter(([name]) => {
      return !name.startsWith("tensorzero::evaluation_name::");
    }),
  );
}

export const formatMetricSummaryValue = (
  value: number,
  metricConfig: MetricConfig,
) => {
  if (metricConfig.type === "boolean") {
    return `${Math.round(value * 100)}%`;
  } else if (metricConfig.type === "float") {
    return value.toFixed(2);
  }
  return value;
};
