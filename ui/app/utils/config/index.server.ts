import { parse } from "smol-toml";
import { promises as fs } from "fs";
import { Config, GatewayConfig } from ".";
import { MetricConfigSchema } from "./metric";
import { RawFunctionConfigSchema } from "./function.server";
import { z } from "zod";
import { EmbeddingModelConfigSchema, ModelConfigSchema } from "./models";
import { ToolConfigSchema } from "./tool";
import type { FunctionConfig } from "./function";

const DEFAULT_CONFIG_PATH = "/app/config/tensorzero.toml";
const ENV_CONFIG_PATH = process.env.TENSORZERO_UI_CONFIG_PATH;

export async function loadConfig(config_path?: string): Promise<Config> {
  // If the config_path was provided (via the env var)
  if (config_path) {
    try {
      // Check if the file exists
      await fs.access(config_path);
    } catch {
      throw new Error(`Configuration file not found at ${config_path}`);
    }
  } else {
    // If the env var is not set, try the default location.
    try {
      await fs.access(DEFAULT_CONFIG_PATH);
      config_path = DEFAULT_CONFIG_PATH;
      console.info(`Found default config at ${DEFAULT_CONFIG_PATH}`);
    } catch {
      console.warn(
        `Config file not found at ${DEFAULT_CONFIG_PATH}. Using blank config.`,
      );
      // Return a blank config if no file is available.
      return {
        gateway: { disable_observability: false },
        models: {},
        embedding_models: {},
        functions: {},
        metrics: {},
        tools: {},
      };
    }
  }

  // At this point, config_path is guaranteed to point to an existing file.
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
    // Pass in ENV_CONFIG_PATH; if not set, loadConfig() will try the default path.
    configPromise = loadConfig(ENV_CONFIG_PATH);
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
