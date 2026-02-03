/**
 * Configuration Context Provider
 *
 * This module provides a React Context for managing the TensorZero configuration object.
 * It exports a ConfigProvider component and a useConfig hook for accessing
 * configuration values throughout the application.
 *
 * @module config
 */

import { createContext, useContext } from "react";
import type { UiConfig } from "~/types/tensorzero";

/**
 * Default empty config used when the gateway is unavailable.
 * Components will see empty lists/objects rather than crashing.
 */
export const EMPTY_CONFIG: UiConfig = {
  functions: {},
  metrics: {},
  tools: {},
  evaluations: {},
  model_names: [],
  config_hash: "",
};

const ConfigContext = createContext<UiConfig>(EMPTY_CONFIG);

export function useConfig(): UiConfig {
  return useContext(ConfigContext);
}

/**
 * Hook to get a specific function configuration by name
 * @param functionName - The name of the function to retrieve
 * @returns The function configuration object or null if not found
 */
export function useFunctionConfig(functionName: string | null) {
  const config = useConfig();
  if (!functionName) {
    return null;
  }
  return config.functions[functionName] || null;
}

/**
 * Hook to get all function configs
 * @returns The function configuration object or null if not found
 */
export function useAllFunctionConfigs() {
  const config = useConfig();
  return config.functions;
}

export function ConfigProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: UiConfig;
}) {
  return (
    <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>
  );
}
