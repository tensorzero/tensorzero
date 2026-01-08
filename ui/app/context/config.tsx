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

const ConfigContext = createContext<UiConfig | undefined>(undefined);

/**
 * Hook to get the TensorZero configuration.
 * Returns undefined when config is unavailable (e.g., gateway down).
 * Components should handle undefined gracefully by showing empty/disabled states.
 */
export function useConfig(): UiConfig | undefined {
  return useContext(ConfigContext);
}

/**
 * Hook to get a specific function configuration by name.
 * Returns null if config unavailable or function not found.
 * @param functionName - The name of the function to retrieve
 * @returns The function configuration object or null
 */
export function useFunctionConfig(functionName: string | null) {
  const config = useConfig();
  if (!config || !functionName) {
    return null;
  }
  // eslint-disable-next-line no-restricted-syntax
  return config.functions[functionName] || null;
}

/**
 * Hook to get all function configs.
 * Returns null if config is unavailable.
 * @returns The functions map or null
 */
export function useAllFunctionConfigs() {
  const config = useConfig();
  if (!config) {
    return null;
  }
  // eslint-disable-next-line no-restricted-syntax
  return config.functions;
}

export function ConfigProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: UiConfig | undefined;
}) {
  return (
    <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>
  );
}
