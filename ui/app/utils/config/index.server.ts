/**
 * Configuration loader for TensorZero UI.
 *
 * The config is loaded from the gateway and cached. The server periodically
 * polls the gateway's status endpoint to check if the config hash has changed.
 * If it has, the cache is invalidated so the next request will load fresh config.
 * This polling happens once for the entire server process, shared across all
 * browser clients.
 */

import { redirect } from "react-router";
import type { FunctionConfig, UiConfig } from "~/types/tensorzero";
import { getTensorZeroClient } from "../get-tensorzero-client.server";
import { DEFAULT_FUNCTION } from "../constants";
import { hexToDecimal } from "../common";
import { getEnv } from "../env.server";
import { logger } from "../logger";

// Poll interval in milliseconds (5 seconds)
const CONFIG_HASH_POLL_INTERVAL_MS = 5_000;

// Track if polling has been started
let pollingStarted = false;

/**
 * Loads config from the gateway.
 */
export async function loadConfig(): Promise<UiConfig> {
  const client = getTensorZeroClient();
  return await client.getUiConfig();
}

/**
 * In cookie-auth deployments (no TENSORZERO_API_KEY), the gateway credential
 * comes from a per-request cookie via AsyncLocalStorage. A process-global
 * cache populated under one request's cookie would (a) leak that user's
 * config to later unauthenticated requests and (b) suppress the 401 the
 * root loader uses to render the API-key dialog. Skip caching in that mode.
 */
function isCookieAuthMode(): boolean {
  return !getEnv().TENSORZERO_API_KEY;
}

let configCache: UiConfig | undefined = undefined;

/**
 * Autopilot UI is hard-disabled. Revert this function (see git history) to
 * restore the gateway status check + TTL cache.
 */
// eslint-disable-next-line @typescript-eslint/require-await
export async function checkAutopilotAvailable(): Promise<boolean> {
  return false;
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
 *
 * No-op in cookie-auth mode: the setInterval tick has no request context, so
 * its `getEffectiveApiKey()` would return undefined and every poll would 401.
 * There is also no shared cache to invalidate in that mode.
 */
function startConfigHashPolling(): void {
  if (pollingStarted || isCookieAuthMode()) {
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
  experimentation: {
    base: { type: "static", candidate_variants: [], fallback_variants: [] },
    namespaces: {},
  },
};

async function loadAndDecorateConfig(): Promise<UiConfig> {
  const freshConfig = await loadConfig();
  // eslint-disable-next-line no-restricted-syntax
  freshConfig.functions[DEFAULT_FUNCTION] = defaultFunctionConfig;
  return freshConfig;
}

/**
 * Gets the config, using the cache if available.
 * Also starts the background polling for config hash changes if not already started.
 *
 * In cookie-auth mode, no caching: each call hits the gateway with the
 * current request's cookie so credential checks aren't bypassed.
 */
export async function getConfig(): Promise<UiConfig> {
  if (isCookieAuthMode()) {
    return loadAndDecorateConfig();
  }

  // Start polling for config hash changes (only starts once)
  startConfigHashPolling();

  // If we have a cached config, return it
  if (configCache) {
    return configCache;
  }

  // Cache doesn't exist or was invalidated, load it.
  configCache = await loadAndDecorateConfig();
  return configCache;
}

/**
 * Clears all cached config state so the next read fetches from the gateway.
 */
export function invalidateConfigCache(): void {
  configCache = undefined;
  snapshotConfigCache.clear();
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
// Snapshot-aware config resolution from request
// ============================================================================

/**
 * Reads `?snapshot_hash` from the request URL and resolves the appropriate
 * config — either the historical snapshot or the current config.
 *
 * If the snapshot hash matches the current config hash (after converting
 * hex→decimal), strips the param via redirect so the URL stays clean.
 * Otherwise resolves the historical snapshot config (falling back to
 * current on error).
 */
export async function getConfigFromRequest(
  request: Request,
): Promise<UiConfig> {
  const url = new URL(request.url);
  const snapshotHash = url.searchParams.get("snapshot_hash");
  if (!snapshotHash) return getConfig();

  const currentConfig = await getConfig();
  const decimalHash = hexToDecimal(snapshotHash);

  // Fast path: if the URL hash matches current config, strip it.
  if (currentConfig.config_hash === decimalHash) {
    url.searchParams.delete("snapshot_hash");
    throw redirect(url.pathname + url.search);
  }

  return getConfigForSnapshot(snapshotHash);
}

// ============================================================================
// Snapshot config fetching
// ============================================================================

const snapshotConfigCache = new Map<string, UiConfig>();
const MAX_SNAPSHOT_CACHE_SIZE = 50;

/**
 * Fetches the config for a given snapshot hash. If the hash matches the current
 * config, returns that. Otherwise fetches the historical config from the gateway
 * and caches it. Falls back to the current config on error.
 */
export async function getConfigForSnapshot(
  snapshotHash: string | undefined | null,
): Promise<UiConfig> {
  if (!snapshotHash) return getConfig();

  const decimalHash = hexToDecimal(snapshotHash);

  const currentConfig = await getConfig();
  if (currentConfig.config_hash === decimalHash) return currentConfig;

  const cached = snapshotConfigCache.get(decimalHash);
  if (cached) return cached;

  try {
    const client = getTensorZeroClient();
    const snapshotConfig = await client.getUiConfigByHash(decimalHash);
    // eslint-disable-next-line no-restricted-syntax
    snapshotConfig.functions[DEFAULT_FUNCTION] = defaultFunctionConfig;

    if (snapshotConfigCache.size >= MAX_SNAPSHOT_CACHE_SIZE) {
      const firstKey = snapshotConfigCache.keys().next().value;
      if (firstKey) snapshotConfigCache.delete(firstKey);
    }
    snapshotConfigCache.set(decimalHash, snapshotConfig);
    return snapshotConfig;
  } catch (error) {
    logger.warn(`Failed to fetch config for snapshot ${snapshotHash}:`, error);
    return currentConfig;
  }
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
  snapshotConfigCache.clear();
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
