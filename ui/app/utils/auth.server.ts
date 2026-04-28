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
import type { UiConfig } from "~/types/tensorzero";
import { getEffectiveApiKey } from "./api-key-override.server";
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
 * No-ops when:
 *  - `[gateway.auth] enabled = false` (the API key table is meaningless), or
 *  - `TENSORZERO_POSTGRES_URL` is unset (page already shows `PostgresRequiredState`).
 */
export async function requireValidApiKeyIfEnabled(
  config: UiConfig,
): Promise<void> {
  if (!config.auth_enabled) return;
  if (!isPostgresAvailable()) return;

  const key = getEffectiveApiKey();
  if (!key) {
    throw data(
      { errorType: InfraErrorType.GatewayAuthFailed },
      { status: 401 },
    );
  }

  try {
    const client = await getPostgresClient();
    await client.validateApiKey(key);
  } catch (error) {
    logger.warn("Rejected request: API key validation failed", error);
    throw data(
      { errorType: InfraErrorType.GatewayAuthFailed },
      { status: 401 },
    );
  }
}
