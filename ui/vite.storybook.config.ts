/**
 * This file exists because Storybook needs a separate Vite configuration that doesn't include
 * the react-router plugin. The react-router plugin is only needed for the main application
 * and would cause issues in Storybook's development environment.
 *
 * Instead of maintaining two separate config files, this file programmatically loads
 * the main vite.config.ts and removes the react-router plugin at runtime.
 */

import { defineConfig } from "vite";
import { loadConfigFromFile } from "vite";
import type { PluginOption, Plugin } from "vite";

// Load the main vite config
const mainConfig = await loadConfigFromFile(
  { command: "serve", mode: "development" },
  "vite.config.ts",
);

if (!mainConfig) {
  throw new Error("Failed to load vite.config.ts");
}

// Remove the react-router plugin and its import
const config = mainConfig.config;
if (config.plugins) {
  config.plugins = config.plugins.filter((plugin: PluginOption) => {
    // The react-router plugin is an array of plugins
    // We need to filter out the array that contains all the react-router plugins
    if (Array.isArray(plugin)) {
      return !plugin.some((p) => (p as Plugin).name === "react-router");
    }
    return true;
  });
}

export default defineConfig(config);
