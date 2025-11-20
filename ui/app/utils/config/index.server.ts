import type { Config, FunctionConfig } from "~/types/tensorzero";
import { getConfig as getConfigNative } from "tensorzero-node";
import { getEnv } from "../env.server";
import { DEFAULT_FUNCTION } from "../constants";

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

export async function loadConfig(): Promise<Config> {
  const env = getEnv();
  if (env.TENSORZERO_UI_DEFAULT_CONFIG) {
    return await getConfigNative(null);
  }
  const config = await getConfigNative(env.TENSORZERO_UI_CONFIG_PATH);
  return config;
}

/**
 * Helper function to get the config path used by the UI.
 * Returns null if using default config, otherwise returns the config path.
 */
export function getConfigPath(): string | null {
  const env = getEnv();
  if (env.TENSORZERO_UI_DEFAULT_CONFIG) {
    return null;
  }
  return env.TENSORZERO_UI_CONFIG_PATH;
}

interface ConfigCache {
  data: Config;
  timestamp: number;
}

let configCache: ConfigCache | null = null;

const defaultFunctionConfig: FunctionConfig = {
  type: "chat",
  variants: {},
  schemas: {},
  tools: [],
  tool_choice: "auto",
  parallel_tool_calls: null,
  description:
    "This is the default function for TensorZero. This function is used when you call a model directly without specifying a function name. It has no variants preconfigured because they are generated dynamically at inference time based on the model being called.",
  experimentation: { type: "uniform" },
};

export function getDefaultFunctionConfigWithVariant(
  model_name: string,
): FunctionConfig {
  const functionConfig = defaultFunctionConfig;
  functionConfig.variants[model_name] = {
    inner: {
      type: "chat_completion",
      model: model_name,
      weight: null,
      templates: {},
      temperature: null,
      top_p: null,
      max_tokens: null,
      presence_penalty: null,
      frequency_penalty: null,
      seed: null,
      stop_sequences: null,
      json_mode: null,
      retries: { num_retries: 0, max_delay_s: 0 },
    },
    timeouts: {
      non_streaming: { total_ms: null },
      streaming: { ttft_ms: null },
    },
  };
  return functionConfig;
}

export async function getConfig() {
  const now = Date.now();

  if (configCache && now - configCache.timestamp < CACHE_TTL_MS) {
    return configCache.data;
  }

  // Cache is invalid or doesn't exist, reload it
  const freshConfig = await loadConfig();
  // eslint-disable-next-line no-restricted-syntax
  freshConfig.functions[DEFAULT_FUNCTION] = defaultFunctionConfig;

  configCache = { data: freshConfig, timestamp: now };
  return freshConfig;
}

/**
 * Helper function to get a specific function configuration by name (server-side only)
 * @param functionName - The name of the function to retrieve
 * @param config - The config object (optional, will fetch if not provided)
 * @returns The function configuration object or null if not found
 */
export async function getFunctionConfig(functionName: string, config?: Config) {
  const cfg = config || (await getConfig());
  // eslint-disable-next-line no-restricted-syntax
  return cfg.functions[functionName] || null;
}

/**
 * Helper function to get all function configurations (server-side only)
 * @param config - The config object (optional, will fetch if not provided)
 * @returns The function configuration object or null if not found
 */
export async function getAllFunctionConfigs(config?: Config) {
  const cfg = config || (await getConfig());

  return cfg.functions;
}
