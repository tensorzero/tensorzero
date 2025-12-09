/**
 * Configuration loader for TensorZero UI.
 *
 * The config for TensorZero UI can be loaded from the gateway or from
 * disk (legacy behavior), and is used by a large number of UI components.
 * This is the server component that loads the config and keeps it fresh.
 *
 * The server periodically polls the gateway's status endpoint to check if the
 * config hash has changed. If it has, the cache is invalidated so the next
 * request will load fresh config. This polling happens once for the entire
 * server process, shared across all browser clients.
 */

import type { Config, FunctionConfig, UiConfig } from "~/types/tensorzero";
import { getTensorZeroClient } from "../get-tensorzero-client.server";
import { getEnv } from "../env.server";
import { DEFAULT_FUNCTION } from "../constants";
import { logger } from "../logger";

// Poll interval in milliseconds (30 seconds)
const CONFIG_HASH_POLL_INTERVAL_MS = 30_000;

// Track if polling has been started
let pollingStarted = false;

/**
 * Converts a full Config (from disk) to a UiConfig (for the UI context).
 * Note: When loading from disk, we don't have a config_hash, so we use an empty string.
 * This is only used in legacy mode (when not using the gateway).
 */
function configToUiConfig(config: Config): UiConfig {
  return {
    // eslint-disable-next-line no-restricted-syntax
    functions: config.functions,
    metrics: config.metrics,
    tools: config.tools,
    evaluations: config.evaluations,
    model_names: Object.keys(config.models.table),
    config_hash: "", // Not available when loading from disk
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

/**
 * Checks if the config hash has changed by polling the gateway's status endpoint.
 * If the hash has changed, invalidates the cache so the next getConfig() call
 * will load fresh config.
 */
async function checkConfigHash(): Promise<void> {
  // Skip if no cached config or no hash (legacy disk mode)
  if (!configCache?.config_hash) {
    return;
  }

  try {
    const status = await getTensorZeroClient().status();
    const gatewayHash = status.config_hash;

    if (gatewayHash !== configCache.config_hash) {
      logger.info(
        `Config hash changed from ${configCache.config_hash} to ${gatewayHash}, invalidating cache`,
      );
      configCache = undefined;
    }
  } catch (error) {
    // Log but don't throw - polling failures shouldn't crash the server
    logger.warn("Failed to check config hash:", error);
  }
}

/**
 * Starts the periodic config hash polling.
 * This is called automatically when getConfig() is first called.
 * The polling runs once for the entire server process.
 */
function startConfigHashPolling(): void {
  if (pollingStarted) {
    return;
  }
  pollingStarted = true;

  // Start polling in the background
  setInterval(() => {
    checkConfigHash().catch((error) => {
      logger.error("Config hash polling error:", error);
    });
  }, CONFIG_HASH_POLL_INTERVAL_MS);

  logger.info(
    `Started config hash polling (interval: ${CONFIG_HASH_POLL_INTERVAL_MS}ms)`,
  );
}

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

/**
 * Gets the config, using the cache if available.
 * Also starts the background polling for config hash changes if not already started.
 */
export async function getConfig(): Promise<UiConfig> {
  // Start polling for config hash changes (only starts once)
  startConfigHashPolling();

  // If we have a cached config, return it
  if (configCache) {
    return configCache;
  }

  // Cache doesn't exist or was invalidated, load it.
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

// ============================================================================
// Testing utilities - exported for testing only
// ============================================================================

/**
 * Resets the module state for testing purposes.
 * @internal This function is only exported for testing.
 */
export function _resetForTesting(): void {
  configCache = undefined;
  pollingStarted = false;
}

/**
 * Manually triggers a config hash check for testing purposes.
 * @internal This function is only exported for testing.
 */
export async function _checkConfigHashForTesting(): Promise<void> {
  return checkConfigHash();
}

/**
 * Gets the current cached config for testing purposes.
 * @internal This function is only exported for testing.
 */
export function _getConfigCacheForTesting(): UiConfig | undefined {
  return configCache;
}

/**
 * Sets the config cache for testing purposes.
 * @internal This function is only exported for testing.
 */
export function _setConfigCacheForTesting(config: UiConfig | undefined): void {
  configCache = config;
}
