import type { Config, FunctionConfig, UiConfig } from "~/types/tensorzero";
import { getTensorZeroClient } from "../get-tensorzero-client.server";
import { getEnv } from "../env.server";
import { DEFAULT_FUNCTION } from "../constants";

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

/**
 * Converts a full Config (from disk) to a UiConfig (for the UI context).
 */
function configToUiConfig(config: Config): UiConfig {
  return {
    // eslint-disable-next-line no-restricted-syntax
    functions: config.functions,
    metrics: config.metrics,
    tools: config.tools,
    evaluations: config.evaluations,
    model_names: Object.keys(config.models.table),
  };
}

export async function loadConfig(): Promise<UiConfig> {
  const env = getEnv();

  // Use gateway if TENSORZERO_FEATURE_FLAG__UI_CONFIG_FROM_GATEWAY is set
  if (env.TENSORZERO_FEATURE_FLAG__UI_CONFIG_FROM_GATEWAY) {
    const client = getTensorZeroClient();
    return await client.getUiConfig();
  }

  // Otherwise use disk loading via tensorzero-node (legacy behavior)
  const { getConfig: getConfigNative } = await import("tensorzero-node");
  let fullConfig: Config;
  if (env.TENSORZERO_UI_DEFAULT_CONFIG) {
    fullConfig = await getConfigNative(null);
  } else {
    fullConfig = await getConfigNative(env.TENSORZERO_UI_CONFIG_PATH);
  }
  return configToUiConfig(fullConfig);
}

/**
 * Helper function to get the config path used by the UI.
 * Returns null if using default config, otherwise returns the config path.
 * @deprecated This function is deprecated and will be removed in a future version.
 * Config is now loaded from the gateway, not from disk.
 */
export function getConfigPath(): string | null {
  const env = getEnv();
  if (env.TENSORZERO_UI_DEFAULT_CONFIG) {
    return null;
  }
  return env.TENSORZERO_UI_CONFIG_PATH;
}

let configCache: UiConfig | undefined = undefined;

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

export async function getConfig(): Promise<UiConfig> {
  if (configCache) {
    return configCache;
  }

  // Cache doesn't exist, load it.
  const freshConfig = await loadConfig();
  // eslint-disable-next-line no-restricted-syntax
  freshConfig.functions[DEFAULT_FUNCTION] = defaultFunctionConfig;

  configCache = freshConfig;
  return configCache;
}

/**
 * Helper function to get a specific function configuration by name (server-side only)
 * @param functionName - The name of the function to retrieve
 * @param config - The config object (optional, will fetch if not provided)
 * @returns The function configuration object or null if not found
 */
export async function getFunctionConfig(
  functionName: string,
  config?: UiConfig,
) {
  const cfg = config || (await getConfig());
  // eslint-disable-next-line no-restricted-syntax
  return cfg.functions[functionName] || null;
}

/**
 * Helper function to get all function configurations (server-side only)
 * @param config - The config object (optional, will fetch if not provided)
 * @returns The function configuration object or null if not found
 */
export async function getAllFunctionConfigs(config?: UiConfig) {
  const cfg = config || (await getConfig());

  return cfg.functions;
}
