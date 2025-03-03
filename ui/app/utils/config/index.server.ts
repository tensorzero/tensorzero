import { parse } from "smol-toml";
import { promises as fs } from "fs";
import { Config, GatewayConfig } from ".";
import { MetricConfigSchema } from "./metric";
import {
  DEFAULT_FUNCTION_NAME,
  getDefaultFunctionWithVariants,
  RawFunctionConfigSchema,
} from "./function.server";
import { z } from "zod";
import { EmbeddingModelConfigSchema, ModelConfigSchema } from "./models";
import { ToolConfigSchema } from "./tool";
import type { FunctionConfig } from "./function";

const DEFAULT_CONFIG_PATH = "config/tensorzero.toml";
const ENV_CONFIG_PATH = process.env.TENSORZERO_UI_CONFIG_PATH;
const CACHE_TTL_MS = 1000 * 60; // 1 minute

/*
Config Context provider:

In general, the config tree for TensorZero is static and can be loaded at startup and then used by any component.
This is good so that we can avoid reading the config from the file system on every request.
Since it is required for a very large number of components, it is also great to avoid drilling it down through nearly all components.

So we implement a context provider that loads the config at the root of the app and makes it available to all components
via the ConfigProvider and the useConfig hook.

However, there is one exception to this static behavior: the default function `tensorzero::default`.
Since the default function can be called with any model and since doing so with essentially creates a new variant,
we must check what variants have been used in the past for this function.

In order to avoid drilling the config through the entire application, we implement a caching mechanism here that is used for context.
We only reload the config (file + database query) if the config is needed (via the hook or a backend helper function getConfig)
and it has not been loaded in the past CACHE_TTL_MS.

This introduces a small liveness issue where the list of variants for the default function is not updated for up toCACHE_TTL_MS
after a new variant is used.

We will likely address this with some form of query library down the line.
*/

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
        `Config file not found at ${DEFAULT_CONFIG_PATH}. Using blank config. Tip: Set the \`TENSORZERO_UI_CONFIG_PATH\` environment variable to use a different path.`,
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

  // Add default function to the config
  loadedConfig.functions = {
    ...loadedConfig.functions,
    [DEFAULT_FUNCTION_NAME]: await getDefaultFunctionWithVariants(),
  };

  return loadedConfig;
}

interface ConfigCache {
  data: Config;
  timestamp: number;
}

let configCache: ConfigCache | null = null;

export async function getConfig() {
  const now = Date.now();

  if (configCache && now - configCache.timestamp < CACHE_TTL_MS) {
    return configCache.data;
  }

  // Cache is invalid or doesn't exist, reload it
  const freshConfig = await loadConfig(ENV_CONFIG_PATH);

  configCache = { data: freshConfig, timestamp: now };
  return freshConfig;
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
