import { z } from "zod";
import { ModelConfigSchema, EmbeddingModelConfigSchema } from "./models";
import { FunctionConfigSchema } from "./function";
import { MetricConfigSchema } from "./metric";
import { ToolConfigSchema } from "./tool";
import { EvaluationConfigSchema } from "./evaluations";

export const ObservabilityConfigSchema = z.object({
  enabled: z.boolean().optional(),
  async_writes: z.boolean().default(false),
});

export const OtlpConfigSchema = z.object({
  traces: z.object({
    enabled: z.boolean().optional(),
  }),
});

export const ExportConfigSchema = z.object({
  otlp: OtlpConfigSchema,
});

export const GatewayConfig = z.object({
  bind_address: z.string().optional(),
  observability: ObservabilityConfigSchema.optional(),
  debug: z.boolean().default(false),
  enable_template_filesystem_access: z.boolean().default(false),
  export: ExportConfigSchema.optional(),
});

export type GatewayConfig = z.infer<typeof GatewayConfig>;

export const Config = z.object({
  gateway: GatewayConfig.optional(),
  models: z.record(z.string(), ModelConfigSchema),
  embedding_models: z
    .record(z.string(), EmbeddingModelConfigSchema)
    .optional()
    .default({}),
  functions: z.record(z.string(), FunctionConfigSchema),
  metrics: z.record(z.string(), MetricConfigSchema),
  tools: z.record(z.string(), ToolConfigSchema).optional().default({}),
  evaluations: z.record(z.string(), EvaluationConfigSchema),
});
export type Config = z.infer<typeof Config>;
