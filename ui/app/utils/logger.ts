/* eslint-disable no-console */
import { isErrorLike } from "~/utils/common";

const LOG_LEVELS = {
  debug: 1,
  info: 2,
  warn: 3,
  error: 4,
} as const;

type LogLevel = keyof typeof LOG_LEVELS;

let hasLoggedInvalidLogLevel = false;

const getLogLevel = (): LogLevel => {
  if (typeof process !== "undefined") {
    const level = process.env.TENSORZERO_UI_LOG_LEVEL?.toLowerCase();
    if (level) {
      if (level in LOG_LEVELS) {
        return level as LogLevel;
      }
      if (!hasLoggedInvalidLogLevel) {
        console.warn(
          `[TensorZero UI] Invalid TENSORZERO_UI_LOG_LEVEL: "${process.env.TENSORZERO_UI_LOG_LEVEL}". Valid values are: debug, info, warn, error. Defaulting to "info".`,
        );
        hasLoggedInvalidLogLevel = true;
      }
    }
  }
  return "info";
};

const currentLevel = getLogLevel();

const shouldLog = (level: LogLevel): boolean => {
  return LOG_LEVELS[level] >= LOG_LEVELS[currentLevel];
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
