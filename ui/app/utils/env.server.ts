import { logger } from "./logger";

// This is the only file in which `process.env` should be accessed directly.

class EnvironmentVariableError extends Error {
  constructor(
    public message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "EnvironmentVariableError";
  }
}

// Note: TENSORZERO_UI_LOG_LEVEL is handled in logger.ts to avoid circular dependencies.
interface Env {
  TENSORZERO_CLICKHOUSE_URL: string;
  TENSORZERO_POSTGRES_URL: string | null;
  TENSORZERO_UI_READ_ONLY: boolean;
  TENSORZERO_GATEWAY_URL: string;
  TENSORZERO_API_KEY: string | null;
  OPENAI_BASE_URL: string | null;
  FIREWORKS_BASE_URL: string | null;
  FIREWORKS_ACCOUNT_ID: string | null;
  TOGETHER_BASE_URL: string | null;
}

let _env: Env | undefined;
let hasLoggedConfigPathDeprecation = false;

/**
 * Use this function to retrieve the environment variables instead of accessing
 * process.env directly. This ensures that required environment variables are
 * only checked in specific call-sites rather than at the module level, allowing
 * for better error handling and testing.
 */
export function getEnv(): Env {
  if (_env) {
    return _env;
  }

  const TENSORZERO_CLICKHOUSE_URL = getClickhouseUrl();
  const TENSORZERO_GATEWAY_URL = process.env.TENSORZERO_GATEWAY_URL;

  // This error is thrown on startup in tensorzero.server.ts
  if (!TENSORZERO_GATEWAY_URL) {
    throw new EnvironmentVariableError(
      "The environment variable `TENSORZERO_GATEWAY_URL` is not set.",
    );
  }

  // Deprecated in 2025.12; can remove in 2026.02+.
  if (
    (process.env.TENSORZERO_UI_CONFIG_PATH ||
      process.env.TENSORZERO_UI_DEFAULT_CONFIG) &&
    !hasLoggedConfigPathDeprecation
  ) {
    logger.warn(
      "Deprecation Warning: The TensorZero UI now reads the configuration from the gateway. The environment variables `TENSORZERO_UI_CONFIG_PATH` and `TENSORZERO_UI_DEFAULT_CONFIG` are deprecated and ignored. You no longer need to mount the configuration onto the UI container.",
    );
    hasLoggedConfigPathDeprecation = true;
  }

  _env = {
    TENSORZERO_CLICKHOUSE_URL,
    TENSORZERO_POSTGRES_URL: process.env.TENSORZERO_POSTGRES_URL || null,
    TENSORZERO_UI_READ_ONLY: process.env.TENSORZERO_UI_READ_ONLY === "1",
    TENSORZERO_GATEWAY_URL,
    FIREWORKS_ACCOUNT_ID: process.env.FIREWORKS_ACCOUNT_ID || null,
    FIREWORKS_BASE_URL: process.env.FIREWORKS_BASE_URL || null,
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL || null,
    TOGETHER_BASE_URL: process.env.TOGETHER_BASE_URL || null,
    TENSORZERO_API_KEY: process.env.TENSORZERO_API_KEY || null,
  };

  return _env;
}

function getClickhouseUrl() {
  const url = process.env.TENSORZERO_CLICKHOUSE_URL;
  if (url) {
    return url;
  }

  throw new EnvironmentVariableError(
    "The environment variable `TENSORZERO_CLICKHOUSE_URL` is not set.",
  );
}
