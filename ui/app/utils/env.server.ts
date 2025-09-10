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

interface Env {
  TENSORZERO_CLICKHOUSE_URL: string;
  TENSORZERO_UI_CONFIG_PATH: string | null;
  TENSORZERO_UI_DEFAULT_CONFIG: boolean;
  TENSORZERO_GATEWAY_URL: string;
  TENSORZERO_EVALUATIONS_PATH: string;
  OPENAI_BASE_URL: string | null;
  FIREWORKS_BASE_URL: string | null;
  FIREWORKS_ACCOUNT_ID: string | null;
}

let _env: Env;

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
  const TENSORZERO_UI_CONFIG_PATH =
    process.env.TENSORZERO_UI_CONFIG_PATH || null;
  const TENSORZERO_UI_DEFAULT_CONFIG =
    (process.env.TENSORZERO_UI_DEFAULT_CONFIG || null) == "1";
  if (!TENSORZERO_UI_CONFIG_PATH && !TENSORZERO_UI_DEFAULT_CONFIG) {
    throw new EnvironmentVariableError(
      "At least one of `TENSORZERO_UI_CONFIG_PATH` or `TENSORZERO_UI_DEFAULT_CONFIG` must be set.",
    );
  }

  const TENSORZERO_GATEWAY_URL = process.env.TENSORZERO_GATEWAY_URL;
  // This error is thrown on startup in tensorzero.server.ts
  if (!TENSORZERO_GATEWAY_URL) {
    throw new EnvironmentVariableError(
      "The environment variable `TENSORZERO_GATEWAY_URL` is not set.",
    );
  }

  _env = {
    TENSORZERO_CLICKHOUSE_URL,
    TENSORZERO_UI_CONFIG_PATH,
    TENSORZERO_UI_DEFAULT_CONFIG,
    TENSORZERO_GATEWAY_URL,
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL || null,
    FIREWORKS_BASE_URL: process.env.FIREWORKS_BASE_URL || null,
    TENSORZERO_EVALUATIONS_PATH:
      process.env.TENSORZERO_EVALUATIONS_PATH || "evaluations",
    FIREWORKS_ACCOUNT_ID: process.env.FIREWORKS_ACCOUNT_ID || null,
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
