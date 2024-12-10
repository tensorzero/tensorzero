import { z } from "zod";
import { ModelConfig, EmbeddingModelConfig } from "./models";
import { parse } from "smol-toml";
import { promises as fs } from "fs";
import { FunctionConfig } from "./function";
import { MetricConfig } from "./metric";
import { ToolConfig } from "./tool";
import path from "path";

export const GatewayConfig = z.object({
  bind_address: z.string().optional(), // Socket address as string
  disable_observability: z.boolean().default(false),
});
export type GatewayConfig = z.infer<typeof GatewayConfig>;

export const Config = z.object({
  gateway: GatewayConfig.optional().default({}),
  models: z.record(z.string(), ModelConfig),
  embedding_models: z
    .record(z.string(), EmbeddingModelConfig)
    .optional()
    .default({}),
  functions: z.record(z.string(), FunctionConfig),
  metrics: z.record(z.string(), MetricConfig),
  tools: z.record(z.string(), ToolConfig).optional().default({}),
});
export type Config = z.infer<typeof Config>;

export async function loadConfig(config_path: string): Promise<Config> {
  console.log(`Loading config from ${config_path}`);
  const tomlContent = await fs.readFile(config_path, "utf-8");
  const parsedConfig = parse(tomlContent);
  const validatedConfig = Config.parse(parsedConfig);

  // Load the templates here for each variant
  for (const [, functionConfig] of Object.entries(validatedConfig.functions)) {
    for (const [, variantConfig] of Object.entries(
      functionConfig.variants || {},
    )) {
      if ("system_template" in variantConfig && variantConfig.system_template) {
        variantConfig.system_template = await fs.readFile(
          getTemplatePath(config_path, variantConfig.system_template),
          "utf-8",
        );
      }
      if ("user_template" in variantConfig && variantConfig.user_template) {
        variantConfig.user_template = await fs.readFile(
          getTemplatePath(config_path, variantConfig.user_template),
          "utf-8",
        );
      }
      if (
        "assistant_template" in variantConfig &&
        variantConfig.assistant_template
      ) {
        variantConfig.assistant_template = await fs.readFile(
          getTemplatePath(config_path, variantConfig.assistant_template),
          "utf-8",
        );
      }
    }
  }
  return validatedConfig;
}

function getTemplatePath(config_path: string, template_path: string) {
  return path.join(path.dirname(config_path), template_path);
}
