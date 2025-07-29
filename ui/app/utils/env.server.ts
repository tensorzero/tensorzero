// This is the only file in which `process.env` should be accessed directly.

import { logger } from "./logger";

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
  TENSORZERO_UI_CONFIG_PATH: string;
  TENSORZERO_GATEWAY_URL: string;
  TENSORZERO_EVALUATIONS_PATH: string;
  OPENAI_BASE_URL: string | null;
  FIREWORKS_BASE_URL: string | null;
  FIREWORKS_ACCOUNT_ID: string | null;
  TENSORZERO_FORCE_CACHE_ON: boolean;
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
  const TENSORZERO_UI_CONFIG_PATH = process.env.TENSORZERO_UI_CONFIG_PATH;
  if (!TENSORZERO_UI_CONFIG_PATH) {
    throw new EnvironmentVariableError(
      "The environment variable `TENSORZERO_UI_CONFIG_PATH` is not set.",
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
    TENSORZERO_GATEWAY_URL,
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL || null,
    FIREWORKS_BASE_URL: process.env.FIREWORKS_BASE_URL || null,
    TENSORZERO_EVALUATIONS_PATH:
      process.env.TENSORZERO_EVALUATIONS_PATH || "evaluations",
    FIREWORKS_ACCOUNT_ID: process.env.FIREWORKS_ACCOUNT_ID || null,
    TENSORZERO_FORCE_CACHE_ON:
      process.env.TENSORZERO_FORCE_CACHE_ON === "true",
  };

  return _env;
}

export function getExtraInferenceOptions(): object {
  if (getEnv().TENSORZERO_FORCE_CACHE_ON) {
    return {
      cache_options: {
        enabled: 'on',
      },
    };
  }
  return {};
}

function getClickhouseUrl() {
  const url = process.env.TENSORZERO_CLICKHOUSE_URL;
  if (url) {
    return url;
  }

  if (process.env.CLICKHOUSE_URL) {
    logger.warn(
      'Deprecation Warning: The environment variable "CLICKHOUSE_URL" has been renamed to "TENSORZERO_CLICKHOUSE_URL" and will be removed in a future version. Please update your environment to use "TENSORZERO_CLICKHOUSE_URL" instead.',
    );
    return process.env.CLICKHOUSE_URL;
  }

  throw new EnvironmentVariableError(
    "The environment variable `TENSORZERO_CLICKHOUSE_URL` is not set.",
  );
}
