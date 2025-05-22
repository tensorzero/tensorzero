import { z } from "zod";

export const MetricConfigTypeSchema = z.enum([
  "boolean",
  "float",
  "comment",
  "demonstration",
]);
export type MetricConfigType = z.infer<typeof MetricConfigTypeSchema>;

export const MetricConfigOptimizeSchema = z.enum(["min", "max"]);
export type MetricConfigOptimize = z.infer<typeof MetricConfigOptimizeSchema>;

export const MetricConfigLevelSchema = z.enum(["inference", "episode"]);
export type MetricConfigLevel = z.infer<typeof MetricConfigLevelSchema>;

export const MetricConfigSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("boolean"),
    optimize: MetricConfigOptimizeSchema,
    level: MetricConfigLevelSchema,
  }),
  z.object({
    type: z.literal("float"),
    optimize: MetricConfigOptimizeSchema,
    level: MetricConfigLevelSchema,
  }),
  z.object({
    type: z.literal("comment"),
  }),
  z.object({
    type: z.literal("demonstration"),
    level: z.literal("inference"),
  }),
]);
export type MetricConfig = z.infer<typeof MetricConfigSchema>;

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
  metrics: Record<string, MetricConfig>,
  level: MetricConfigLevel,
): Record<string, MetricConfig> {
  // First filter the metrics by level
  const filteredEntries = Object.entries(metrics).filter(([, metric]) => {
    // The demonstration and comment configs need special handling because they don't have a level
    if (metric.type === "demonstration") {
      return metric.level === level;
    }
    if (metric.type === "comment") {
      return true;
    }
    return metric.level === level;
  });

  // Then sort to put demonstration and comment at the top
  filteredEntries.sort(([, metricA], [, metricB]) => {
    if (metricA.type === "demonstration" || metricA.type === "comment") {
      return -1;
    }
    if (metricB.type === "demonstration" || metricB.type === "comment") {
      return 1;
    }
    return 0;
  });

  return Object.fromEntries(filteredEntries);
}

// Removes metrics that are part of a static evaluation
// These will have names that start with "tensorzero::evaluation_name::"
export function filterStaticEvaluationMetrics(
  metrics: Record<string, MetricConfig>,
) {
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
