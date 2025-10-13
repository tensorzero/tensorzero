/* eslint-disable @typescript-eslint/no-explicit-any */
export const logger = {
  info: (message: any, ...args: any[]) => {
    // `messageStr` is a hack until we figure out a way to enforce the type checker for errors
    const messageStr =
      typeof message === "string" ? message : JSON.stringify(message);
    console.log(`[TensorZero UI ${__APP_VERSION__}] ${messageStr}`, ...args);
  },
  error: (message: any, ...args: any[]) => {
    // `messageStr` is a hack until we figure out a way to enforce the type checker for errors
    const messageStr =
      typeof message === "string" ? message : JSON.stringify(message);

    console.error(`[TensorZero UI ${__APP_VERSION__}] ${messageStr}`, ...args);
  },
  warn: (message: any, ...args: any[]) => {
    // `messageStr` is a hack until we figure out a way to enforce the type checker for errors
    const messageStr =
      typeof message === "string" ? message : JSON.stringify(message);

    console.warn(`[TensorZero UI ${__APP_VERSION__}] ${messageStr}`, ...args);
  },
  debug: (message: any, ...args: any[]) => {
    // `messageStr` is a hack until we figure out a way to enforce the type checker for errors
    const messageStr =
      typeof message === "string" ? message : JSON.stringify(message);

    console.debug(`[TensorZero UI ${__APP_VERSION__}] ${messageStr}`, ...args);
  },
};
