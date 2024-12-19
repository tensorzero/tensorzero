import { z } from "zod";
import { ModelConfig, EmbeddingModelConfig } from "./models";
import { parse } from "smol-toml";
import { promises as fs } from "fs";
import { FunctionConfig, RawFunctionConfig } from "./function";
import { MetricConfig } from "./metric";
import { ToolConfig } from "./tool";

export const GatewayConfig = z.object({
  bind_address: z.string().optional(), // Socket address as string
  disable_observability: z.boolean().default(false),
});
export type GatewayConfig = z.infer<typeof GatewayConfig>;

export const RawConfig = z
  .object({
    gateway: GatewayConfig.optional().default({}),
    models: z.record(z.string(), ModelConfig),
    embedding_models: z
      .record(z.string(), EmbeddingModelConfig)
      .optional()
      .default({}),
    functions: z.record(z.string(), RawFunctionConfig),
    metrics: z.record(z.string(), MetricConfig),
    tools: z.record(z.string(), ToolConfig).optional().default({}),
  })
  .transform((raw) => {
    const config = { ...raw };
    return {
      ...config,
      load: async function (config_path: string): Promise<Config> {
        const loadedFunctions: Record<string, FunctionConfig> = {};
        for (const [key, func] of Object.entries(config.functions)) {
          loadedFunctions[key] = await func.load(config_path);
        }
        return {
          gateway: config.gateway,
          models: config.models,
          embedding_models: config.embedding_models,
          functions: loadedFunctions,
          metrics: config.metrics,
          tools: config.tools,
        };
      },
    };
  });
export type RawConfig = z.infer<typeof RawConfig>;

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
  const tomlContent = await fs.readFile(config_path, "utf-8");
  const parsedConfig = parse(tomlContent);
  const validatedConfig = RawConfig.parse(parsedConfig);

  const loadedConfig = await validatedConfig.load(config_path);

  // Load the templates here for each variant
  // for (const [, functionConfig] of Object.entries(validatedConfig.functions)) {
  //   for (const [variantName, variantConfig] of Object.entries(
  //     functionConfig.variants || {}
  //   )) {
  //     if (variantConfig.type === "chat_completion") {
  //       // Transform the variant into ChatCompletionConfigWithTemplates
  //       const templatedVariant = variantConfig as ChatCompletionConfig;

  //       if (
  //         "system_template" in variantConfig &&
  //         variantConfig.system_template
  //       ) {
  //         const content = await fs.readFile(
  //           getTemplatePath(config_path, variantConfig.system_template),
  //           "utf-8"
  //         );
  //         templatedVariant.system_template = {
  //           path: variantConfig.system_template,
  //           content,
  //         };
  //       }

  //       if ("user_template" in variantConfig && variantConfig.user_template) {
  //         const content = await fs.readFile(
  //           getTemplatePath(config_path, variantConfig.user_template),
  //           "utf-8"
  //         );
  //         templatedVariant.user_template = {
  //           path: variantConfig.user_template,
  //           content,
  //         };
  //       }

  //       if (
  //         "assistant_template" in variantConfig &&
  //         variantConfig.assistant_template
  //       ) {
  //         const content = await fs.readFile(
  //           getTemplatePath(config_path, variantConfig.assistant_template),
  //           "utf-8"
  //         );
  //         templatedVariant.assistant_template = {
  //           path: variantConfig.assistant_template,
  //           content,
  //         };
  //       }

  //       // Replace the original variant with the templated version
  //       functionConfig.variants[variantName] = templatedVariant;
  //     }
  //   }
  // }
  return loadedConfig;
}

// function getTemplatePath(config_path: string, template_path: string) {
//   return path.join(path.dirname(config_path), template_path);
// }
