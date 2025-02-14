import { parse } from "smol-toml";
import { promises as fs } from "fs";
import { Config, GatewayConfig } from ".";
import { MetricConfigSchema } from "./metric";
import { RawFunctionConfigSchema } from "./function.server";
import { z } from "zod";
import { EmbeddingModelConfigSchema, ModelConfigSchema } from "./models";
import { ToolConfigSchema } from "./tool";
import type { FunctionConfig } from "./function";
import path from "path";


function checkQuotedEnvVar(key: string, value: string | undefined): void {
  if (!value) return;
  
  const isQuoted = (value.startsWith('"') && value.endsWith('"')) || 
                   (value.startsWith("'") && value.endsWith("'"));
  
  if (isQuoted) {
    console.warn(
      `Warning: Environment variable ${key} contains quotes. ` +
      `This may cause issues. Remove the quotes from the value in your .env file.`
    );
  }
}

function validateEnvironmentVariables(requiredVars: string[]): void {
  for (const varName of requiredVars) {
    const value = process.env[varName];
    checkQuotedEnvVar(varName, value);
  }
}


const CONFIG_PATH =
  process.env.TENSORZERO_UI_CONFIG_PATH ||
  path.join("config", "tensorzero.toml");

export async function loadConfig(config_path: string): Promise<Config> {

  
    validateEnvironmentVariables([
    'TENSORZERO_UI_CONFIG_PATH',
    'TENSORZERO_GATEWAY_URL',
    'TENSORZERO_CLICKHOUSE_URL',
    'OPENAI_API_KEY',
    'FIREWORKS_API_KEY',
    'FIREWORKS_ACCOUNT_ID'
  ]);

  const tomlContent = await fs.readFile(config_path, "utf-8");
  const parsedConfig = parse(tomlContent);
  const validatedConfig = RawConfig.parse(parsedConfig);

  const loadedConfig = await validatedConfig.load(config_path);

  // Add demonstration metric to the config
  loadedConfig.metrics = {
    ...loadedConfig.metrics,
    demonstration: {
      type: "demonstration" as const,
      level: "inference" as const,
    },
  };

  return loadedConfig;
}

// Create singleton
let configPromise: ReturnType<typeof loadConfig>;

export function getConfig() {
  if (!configPromise) {
    configPromise = loadConfig(CONFIG_PATH);
  }
  return configPromise;
}

export const RawConfig = z
  .object({
    gateway: GatewayConfig.optional().default({}),
    models: z.record(z.string(), ModelConfigSchema).optional().default({}),
    embedding_models: z
      .record(z.string(), EmbeddingModelConfigSchema)
      .optional()
      .default({}),
    functions: z
      .record(z.string(), RawFunctionConfigSchema)
      .optional()
      .default({}),
    metrics: z.record(z.string(), MetricConfigSchema).optional().default({}),
    tools: z.record(z.string(), ToolConfigSchema).optional().default({}),
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
