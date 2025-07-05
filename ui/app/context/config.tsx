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
import type { Config } from "tensorzero-node";

const ConfigContext = createContext<Config | null>(null);

export function useConfig() {
  const config = useContext(ConfigContext);
  if (!config) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  return config;
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
