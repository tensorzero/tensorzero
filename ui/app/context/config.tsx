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
import type { Config } from "~/types/tensorzero";

const ConfigContext = createContext<Config | null>(null);

export function useConfig() {
  const config = useContext(ConfigContext);
  if (!config) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  return config;
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
  // eslint-disable-next-line no-restricted-syntax
  return config.functions[functionName] || null;
}
/**
 * Hook to get all function configs
 * @returns The function configuration object or null if not found
 */
export function useAllFunctionConfigs() {
  const config = useConfig();
  // eslint-disable-next-line no-restricted-syntax
  return config.functions;
}

export function ConfigProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: Config;
}) {
  return (
    <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>
  );
}
