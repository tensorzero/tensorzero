import { z } from "zod";
import { ModelConfigSchema, EmbeddingModelConfigSchema } from "./models";
import { FunctionConfigSchema } from "./function";
import { MetricConfigSchema } from "./metric";
import { ToolConfigSchema } from "./tool";
import { EvalConfigSchema } from "./evals";

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
  evals: z.record(z.string(), EvalConfigSchema),
});
export type Config = z.infer<typeof Config>;
