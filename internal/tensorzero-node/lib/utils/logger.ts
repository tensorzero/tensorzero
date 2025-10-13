/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Creates a logger instance with a custom prefix
 * @param prefix - The prefix to use for all log messages (e.g., "TensorZero Node", "TensorZero UI 1.0.0")
 * @returns Logger instance with error, warn, info, and debug methods
 */
export function createLogger(prefix: string) {
  return {
    error: (message: any, ...args: any[]) => {
      const messageStr =
        typeof message === "string" ? message : JSON.stringify(message);
      console.error(`[${prefix}] ${messageStr}`, ...args);
    },
    warn: (message: any, ...args: any[]) => {
      const messageStr =
        typeof message === "string" ? message : JSON.stringify(message);
      console.warn(`[${prefix}] ${messageStr}`, ...args);
    },
    info: (message: any, ...args: any[]) => {
      const messageStr =
        typeof message === "string" ? message : JSON.stringify(message);
      console.log(`[${prefix}] ${messageStr}`, ...args);
    },
    debug: (message: any, ...args: any[]) => {
      const messageStr =
        typeof message === "string" ? message : JSON.stringify(message);
      console.debug(`[${prefix}] ${messageStr}`, ...args);
    },
  };
}

/**
 * Default logger instance for tensorzero-node internal use
 */
export const logger = createLogger("TensorZero Node");
