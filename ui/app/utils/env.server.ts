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
  TENSORZERO_POSTGRES_URL?: string;
  TENSORZERO_UI_READ_ONLY: boolean;
  TENSORZERO_GATEWAY_URL: string;
  TENSORZERO_API_KEY?: string;
  TENSORZERO_UI_CONFIG_FILE?: string;
  TENSORZERO_AUTOPILOT_BETA_TOOLS?: string;
  autopilotHeaders: Record<string, string>;
}

let _env: Env | undefined;
let hasLoggedConfigPathDeprecation = false;
let hasLoggedClickhouseUrlDeprecation = false;

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

  // Deprecated in 2025.12; can remove in 2026.02+.
  if (
    process.env.TENSORZERO_CLICKHOUSE_URL &&
    !hasLoggedClickhouseUrlDeprecation
  ) {
    logger.warn(
      "Deprecation Warning: TensorZero UI now makes all database queries through the gateway. The environment variable `TENSORZERO_CLICKHOUSE_URL` is deprecated and ignored.",
    );
    hasLoggedClickhouseUrlDeprecation = true;
  }

  // Collect TENSORZERO_HEADER_* env vars as autopilot headers.
  // e.g. TENSORZERO_HEADER_BETA_TOOLS=value -> tensorzero-beta-tools: value
  const autopilotHeaders: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key.startsWith("TENSORZERO_HEADER_") && value) {
      const headerName =
        "tensorzero-" +
        key.slice("TENSORZERO_HEADER_".length).toLowerCase().replace(/_/g, "-");
      autopilotHeaders[headerName] = value;
    }
  }

  _env = {
    TENSORZERO_POSTGRES_URL: process.env.TENSORZERO_POSTGRES_URL,
    TENSORZERO_UI_READ_ONLY: process.env.TENSORZERO_UI_READ_ONLY === "1",
    TENSORZERO_GATEWAY_URL,
    TENSORZERO_API_KEY: process.env.TENSORZERO_API_KEY,
    TENSORZERO_UI_CONFIG_FILE: process.env.TENSORZERO_UI_CONFIG_FILE,
    TENSORZERO_AUTOPILOT_BETA_TOOLS:
      process.env.TENSORZERO_AUTOPILOT_BETA_TOOLS,
    autopilotHeaders,
  };

  return _env;
}
