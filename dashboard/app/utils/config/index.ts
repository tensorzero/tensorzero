import { z } from "zod";
import { ModelConfig, EmbeddingModelConfig } from "./models";
import { parse } from "smol-toml";
import { promises as fs } from "fs";
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

export async function loadConfig(config_path: string): Promise<Config> {
  const tomlContent = await fs.readFile(config_path, "utf-8");
  const parsedConfig = parse(tomlContent);
  const validatedConfig = Config.parse(parsedConfig);
  return validatedConfig;
}
