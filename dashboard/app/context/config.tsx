import { createContext, useContext } from "react";
import { Config } from "~/utils/config";

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
