import { loadConfig } from "~/utils/config";

const CONFIG_PATH =
  process.env.CONFIG_PATH || "fixtures/config/tensorzero.toml";

// Create singleton
let configPromise: ReturnType<typeof loadConfig>;

export function getConfig() {
  if (!configPromise) {
    configPromise = loadConfig(CONFIG_PATH);
  }
  return configPromise;
}
