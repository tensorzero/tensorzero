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
    level: MetricConfigLevelSchema,
  }),
  z.object({
    type: z.literal("demonstration"),
    level: z.literal("inference"),
  }),
]);
export type MetricConfig = z.infer<typeof MetricConfigSchema>;
