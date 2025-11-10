/* eslint-disable no-console */
import { isErrorLike } from "~/utils/common";

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
    console.info(getErrorMessage(message), ...args);
  },
  error: (message: unknown, ...args: unknown[]) => {
    console.error(getErrorMessage(message), ...args);
  },
  warn: (messageString: unknown, ...args: unknown[]) => {
    const message = getErrorMessage(messageString);
    console.warn(getErrorMessage(message), ...args);
  },
  debug: (message: unknown, ...args: unknown[]) => {
    console.debug(getErrorMessage(message), ...args);
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
