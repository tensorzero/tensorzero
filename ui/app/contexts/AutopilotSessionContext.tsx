import { createContext, useContext, useMemo, type ReactNode } from "react";
import { useLocalStorage } from "~/hooks/use-local-storage";

interface AutopilotSessionContextValue {
  /** Whether YOLO mode (auto-approve all tool calls) is enabled */
  yoloMode: boolean;
  /** Toggle YOLO mode on/off */
  setYoloMode: (value: boolean) => void;
}

const AutopilotSessionContext =
  createContext<AutopilotSessionContextValue | null>(null);

interface AutopilotSessionProviderProps {
  children: ReactNode;
}

export function AutopilotSessionProvider({
  children,
}: AutopilotSessionProviderProps) {
  const [yoloMode, setYoloMode] = useLocalStorage<boolean>(
    "tensorzero-yolo-mode",
    false,
  );

  const value = useMemo(
    () => ({ yoloMode, setYoloMode }),
    [yoloMode, setYoloMode],
  );

  return (
    <AutopilotSessionContext.Provider value={value}>
      {children}
    </AutopilotSessionContext.Provider>
  );
}

export function useAutopilotSession(): AutopilotSessionContextValue {
  const context = useContext(AutopilotSessionContext);
  if (!context) {
    throw new Error(
      "useAutopilotSession must be used within an AutopilotSessionProvider",
    );
  }
  return context;
}
