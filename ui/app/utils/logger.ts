import { createLogger } from "tensorzero-node";

export const logger = createLogger(`TensorZero UI ${__APP_VERSION__}`);

export const getVersionInfo = () => ({
  version: __APP_VERSION__,
});
