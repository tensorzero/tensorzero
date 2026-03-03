/**
 * Shared module-level API key override.
 *
 * When the UI is deployed without TENSORZERO_API_KEY but the gateway
 * requires auth, users can enter a key in the browser. That key is
 * stored here so getTensorZeroClient() and getAutopilotClient() can
 * use it as a fallback.
 *
 * The key lives only in process memory — it dies on server restart.
 * This is designed for single-user deployments; in multi-user setups
 * the last key submitted wins for all users.
 */

import { getEnv } from "./env.server";

let _apiKeyOverride: string | undefined;
let _version = 0;

export function setApiKeyOverride(key: string): void {
  _apiKeyOverride = key;
  _version++;
}

/** Returns the env var if set, otherwise the browser-entered override. */
export function getEffectiveApiKey(): string | undefined {
  return getEnv().TENSORZERO_API_KEY ?? _apiKeyOverride;
}

/** Monotonically increasing counter so consumers can detect changes. */
export function getApiKeyOverrideVersion(): number {
  return _version;
}
