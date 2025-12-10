/**
 * Configuration loader for TensorZero UI.
 *
 * The config is loaded from the gateway and cached. The server periodically
 * polls the gateway's status endpoint to check if the config hash has changed.
 * If it has, the cache is invalidated so the next request will load fresh config.
 * This polling happens once for the entire server process, shared across all
 * browser clients.
 */

import type { FunctionConfig, UiConfig } from "~/types/tensorzero";
import { getTensorZeroClient } from "../get-tensorzero-client.server";
import { DEFAULT_FUNCTION } from "../constants";
import { logger } from "../logger";

// Poll interval in milliseconds (30 seconds)
const CONFIG_HASH_POLL_INTERVAL_MS = 30_000;

// Track if polling has been started
let pollingStarted = false;

/**
 * Loads config from the gateway.
 */
export async function loadConfig(): Promise<UiConfig> {
  const client = getTensorZeroClient();
  return await client.getUiConfig();
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
