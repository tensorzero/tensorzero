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

export const MetricConfigSchema = z.object({
  type: MetricConfigTypeSchema,
  optimize: MetricConfigOptimizeSchema,
  level: MetricConfigLevelSchema,
});
export type MetricConfig = z.infer<typeof MetricConfigSchema>;
