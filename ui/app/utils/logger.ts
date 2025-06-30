/* eslint-disable @typescript-eslint/no-explicit-any */
export const logger = {
  info: (message: string, ...args: any[]) => {
    console.log(`[TensorZero UI v${__APP_VERSION__}] ${message}`, ...args);
  },
  error: (message: any, ...args: any[]) => {
    console.error(`[TensorZero UI v${__APP_VERSION__}] ${message}`, ...args);
  },
  warn: (message: string, ...args: any[]) => {
    console.warn(`[TensorZero UI v${__APP_VERSION__}] ${message}`, ...args);
  },
  debug: (message: string, ...args: any[]) => {
    console.debug(`[TensorZero UI v${__APP_VERSION__}] ${message}`, ...args);
  },
};

export const getVersionInfo = () => ({
  version: __APP_VERSION__,
});
