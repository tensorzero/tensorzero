import { z } from "zod";
import { ModelConfigSchema, EmbeddingModelConfigSchema } from "./models";
import { FunctionConfigSchema, type FunctionConfig } from "./function";
import { MetricConfigSchema, type MetricConfig } from "./metric";
import { ToolConfigSchema, type ToolConfig } from "./tool";

export const GatewayConfig = z.object({
  bind_address: z.string().optional(), // Socket address as string
  disable_observability: z.boolean().default(false),
});
export type GatewayConfig = z.infer<typeof GatewayConfig>;

export const Config = z.object({
  gateway: GatewayConfig.optional().default({}),
  models: z.record(z.string(), ModelConfigSchema),
  embedding_models: z
    .record(z.string(), EmbeddingModelConfigSchema)
    .optional()
    .default({}),
  functions: z.record(z.string(), FunctionConfigSchema),
  metrics: z.record(z.string(), MetricConfigSchema),
  tools: z.record(z.string(), ToolConfigSchema).optional().default({}),
});
export type Config = z.infer<typeof Config>;
