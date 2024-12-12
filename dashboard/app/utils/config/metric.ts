import { z } from "zod";

export const MetricConfigType = z.enum([
  "boolean",
  "float",
  "comment",
  "demonstration",
]);
export type MetricConfigType = z.infer<typeof MetricConfigType>;

export const MetricConfigOptimize = z.enum(["min", "max"]);
export type MetricConfigOptimize = z.infer<typeof MetricConfigOptimize>;

export const MetricConfigLevel = z.enum(["inference", "episode"]);
export type MetricConfigLevel = z.infer<typeof MetricConfigLevel>;

export const MetricConfig = z.object({
  type: MetricConfigType,
  optimize: MetricConfigOptimize,
  level: MetricConfigLevel,
});
export type MetricConfig = z.infer<typeof MetricConfig>;
