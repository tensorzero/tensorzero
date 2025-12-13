/* eslint-disable no-console */
import { isErrorLike } from "~/utils/common";

const LOG_LEVELS = ["debug", "info", "warn", "error"] as const;
type LogLevel = (typeof LOG_LEVELS)[number];

let cachedLogLevel: LogLevel | null = null;
let hasLoggedInvalidLogLevel = false;

const LOG_LEVEL_VALUES: Record<LogLevel, number> = {
  debug: 1,
  info: 2,
  warn: 3,
  error: 4,
};

const getLogLevel = (): LogLevel => {
  if (cachedLogLevel !== null) {
    return cachedLogLevel;
  }
  const level = process.env.TENSORZERO_UI_LOG_LEVEL?.toLowerCase();
  if (level) {
    if ((LOG_LEVELS as readonly string[]).includes(level)) {
      cachedLogLevel = level as LogLevel;
      return cachedLogLevel;
    }
    if (!hasLoggedInvalidLogLevel) {
      console.warn(
        `[TensorZero UI] Invalid TENSORZERO_UI_LOG_LEVEL: "${process.env.TENSORZERO_UI_LOG_LEVEL}". Valid values are: debug, info, warn, error. Defaulting to "info".`,
      );
      hasLoggedInvalidLogLevel = true;
    }
  }
  cachedLogLevel = "info";
  return cachedLogLevel;
};

const shouldLog = (level: LogLevel): boolean => {
  return LOG_LEVEL_VALUES[level] >= LOG_LEVEL_VALUES[getLogLevel()];
};

const APP_VERSION = (() => {
  if (typeof __APP_VERSION__ === "string") {
    return __APP_VERSION__;
  }

  if (typeof process !== "undefined") {
    return process.env.npm_package_version;
  }

  return null;
})();

export const logger = {
  info: (message: unknown, ...args: unknown[]) => {
    if (shouldLog("info")) console.info(getErrorMessage(message), ...args);
  },
  error: (message: unknown, ...args: unknown[]) => {
    if (shouldLog("error")) console.error(getErrorMessage(message), ...args);
  },
  warn: (message: unknown, ...args: unknown[]) => {
    if (shouldLog("warn")) console.warn(getErrorMessage(message), ...args);
  },
  debug: (message: unknown, ...args: unknown[]) => {
    if (shouldLog("debug")) console.debug(getErrorMessage(message), ...args);
  },
};

const getErrorMessage = (error: unknown): string => {
  const prefix = APP_VERSION ? `[TensorZero UI ${APP_VERSION}]` : null;
  let messageString: string | null = null;
  try {
    messageString =
      error == null || error === false
        ? null
        : typeof error === "string"
          ? error
          : isErrorLike(error)
            ? error.message
            : JSON.stringify(error);
  } catch {
    // ignore JSON stringify errors. Non-serializable values should only be
    // passed to the logger as args after the message.
  }
  return [prefix, messageString].filter((v) => v != null).join(" ");
};
