/**
 * Configuration loader for TensorZero UI.
 *
 * The config is loaded from the gateway and cached. The server periodically
 * polls the gateway's status endpoint to check if the config hash has changed.
 * If it has, the cache is invalidated so the next request will load fresh config.
 * This polling happens once for the entire server process, shared across all
 * browser clients.
 */

import type {
  EvaluationConfig,
  FunctionConfig,
  UiConfig,
} from "~/types/tensorzero";
import { getTensorZeroClient } from "../get-tensorzero-client.server";
import { DEFAULT_FUNCTION } from "../constants";
import { logger } from "../logger";

// Poll interval in milliseconds (5 seconds)
const CONFIG_HASH_POLL_INTERVAL_MS = 5_000;

// TTL for autopilot availability cache in milliseconds (30 seconds)
const AUTOPILOT_CACHE_TTL_MS = 30_000;

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
let autopilotAvailableCache: { value: boolean; timestamp: number } | undefined =
  undefined;

/**
 * Checks if autopilot is available by querying the gateway's autopilot status endpoint.
 * This endpoint checks if TENSORZERO_AUTOPILOT_API_KEY is configured on the gateway
 * without making any database queries or pinging the autopilot server.
 * The result is cached with a TTL.
 */
export async function checkAutopilotAvailable(): Promise<boolean> {
  // Check if cache is valid (exists and not expired)
  if (
    autopilotAvailableCache !== undefined &&
    Date.now() - autopilotAvailableCache.timestamp < AUTOPILOT_CACHE_TTL_MS
  ) {
    return autopilotAvailableCache.value;
  }

  try {
    const client = getTensorZeroClient();
    const status = await client.getAutopilotStatus();
    autopilotAvailableCache = { value: status.enabled, timestamp: Date.now() };
    return status.enabled;
  } catch (error) {
    // For network errors, assume unavailable but don't cache
    logger.warn("Failed to check autopilot status:", error);
    return false;
  }
}

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
      logger.debug(
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

  logger.debug(
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
  experimentation: { base: { type: "uniform" }, namespaces: {} },
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
 * Helper function to get all function configurations (server-side only)
 * @param config - The config object (optional, will fetch if not provided)
 * @returns The function configuration object or null if not found
 */
export async function getAllFunctionConfigs(config?: UiConfig) {
  const cfg = config || (await getConfig());

  return cfg.functions;
}

interface ResolvedConfigEntry<T> {
  value: T;
  config: UiConfig;
}

/**
 * Resolves a config entry with automatic retry on cache miss.
 *
 * When autopilot (or manual config changes) creates new entities, the UI's
 * cached config may not include them yet. This function tries the cached config
 * first, and if the accessor returns null/undefined, invalidates the cache,
 * refetches from the gateway, and retries once.
 *
 * @param accessor - Function that extracts the desired entry from a UiConfig
 * @returns The resolved entry and the config it came from, or null if not found even after retry
 */
export async function resolveConfigEntry<T>(
  accessor: (config: UiConfig) => T | null | undefined,
): Promise<ResolvedConfigEntry<T> | null> {
  const config = await getConfig();
  const value = accessor(config);
  if (value != null) return { value, config };

  // Cache miss — invalidate and retry once with fresh config from gateway
  logger.debug("Config entry not found in cache, retrying with fresh config");
  configCache = undefined;
  const freshConfig = await getConfig();
  const freshValue = accessor(freshConfig);
  if (freshValue != null) return { value: freshValue, config: freshConfig };

  return null;
}

/**
 * Resolves a function config by name with retry on cache miss.
 */
export async function resolveFunctionConfig(
  functionName: string,
): Promise<ResolvedConfigEntry<FunctionConfig> | null> {
  // eslint-disable-next-line no-restricted-syntax
  return resolveConfigEntry((cfg) => cfg.functions[functionName]);
}

/**
 * Resolves an evaluation config by name with retry on cache miss.
 */
export async function resolveEvaluationConfig(
  evaluationName: string,
): Promise<ResolvedConfigEntry<EvaluationConfig> | null> {
  return resolveConfigEntry((cfg) => cfg.evaluations[evaluationName]);
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
  autopilotAvailableCache = undefined;
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
