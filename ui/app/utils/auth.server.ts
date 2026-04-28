/**
 * UI auth guard for routes that bypass the gateway.
 *
 * Most pages call the gateway and inherit its auth middleware. Pages that talk
 * directly to Postgres via the NAPI binding (e.g. the API keys page) must
 * validate the caller's key themselves — this helper reuses the same Postgres
 * lookup the gateway middleware uses (`tensorzero_auth::postgres::check_key`,
 * exposed via `PostgresClient.validateApiKey`).
 */

import { data } from "react-router";
import { getEffectiveApiKey } from "./api-key-override.server";
import { getConfig } from "./config/index.server";
import { logger } from "./logger";
import { getPostgresClient, isPostgresAvailable } from "./postgres.server";
import { InfraErrorType } from "./tensorzero/errors";

/**
 * Throws an `InfraErrorType.GatewayAuthFailed` 401 response when the deployment
 * has gateway auth enabled and the request does not carry a valid API key.
 *
 * The thrown shape is recognized by `LayoutErrorBoundary`, which renders the
 * existing "Authentication Required" dialog — the same UX the rest of the UI
 * gets from gateway 401s.
 *
 * No-ops when `TENSORZERO_POSTGRES_URL` is unset (page already shows
 * `PostgresRequiredState`) or when `[gateway.auth] enabled = false`.
 *
 * Order matters: when a key is present we validate it directly against
 * Postgres before touching the gateway. In `gateway_auth_with_browser_key`
 * mode the UI server has no API key of its own, so a `getConfig()` fetch on
 * the cold path would 401 against the gateway and surface as a 500 — running
 * this guard first short-circuits that. Only when no key is present do we
 * fall back to inspecting `auth_enabled`, and a config fetch failing there
 * is itself treated as evidence that auth is enabled.
 *
 * Infrastructure errors (Postgres unavailable, query failures) propagate
 * unchanged so they aren't masked as auth failures.
 */
export async function requireValidApiKeyIfEnabled(): Promise<void> {
  if (!isPostgresAvailable()) return;

  const key = getEffectiveApiKey();

  if (key) {
    const client = await getPostgresClient();
    const result = await client.validateApiKey(key);
    if (result.type === "valid") return;
    logger.warn(`Rejected request: API key validation failed (${result.type})`);
    throw data(
      { errorType: InfraErrorType.GatewayAuthFailed },
      { status: 401 },
    );
  }

  // No key — only reject if the gateway is enforcing auth. If the config
  // fetch itself fails (most plausibly because the gateway demands auth that
  // we cannot supply), treat that as confirmation that auth is enabled.
  let authEnabled: boolean;
  try {
    const config = await getConfig();
    authEnabled = config.auth_enabled;
  } catch (error) {
    logger.warn(
      "Failed to load UI config while checking auth state; assuming auth is enabled",
      error,
    );
    authEnabled = true;
  }

  if (!authEnabled) return;

  throw data({ errorType: InfraErrorType.GatewayAuthFailed }, { status: 401 });
}
