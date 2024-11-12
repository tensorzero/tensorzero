import { z } from "zod";
import { ModelConfig, EmbeddingModelConfig } from "./models";
import { FunctionConfig } from "./function";
import { MetricConfig } from "./metric";
import { ToolConfig } from "./tool";

export const GatewayConfig = z.object({
  bind_address: z.string().optional(), // Socket address as string
  disable_observability: z.boolean().default(false),
});
export type GatewayConfig = z.infer<typeof GatewayConfig>;

export const Config = z.object({
  gateway: GatewayConfig,
  models: z.record(z.string(), ModelConfig),
  embedding_models: z.record(z.string(), EmbeddingModelConfig),
  functions: z.record(z.string(), FunctionConfig),
  metrics: z.record(z.string(), MetricConfig),
  tools: z.record(z.string(), ToolConfig),
});
export type Config = z.infer<typeof Config>;
