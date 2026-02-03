import { createContext, useContext, type ReactNode } from "react";
import { useLocalStorage } from "~/hooks/use-local-storage";

interface AutopilotSessionContextValue {
  yoloMode: boolean;
  setYoloMode: (value: boolean) => void;
}

const AutopilotSessionContext =
  createContext<AutopilotSessionContextValue | null>(null);

export function AutopilotSessionProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [yoloMode, setYoloMode] = useLocalStorage<boolean>(
    "tensorzero-yolo-mode",
    false,
  );

  return (
    <AutopilotSessionContext.Provider value={{ yoloMode, setYoloMode }}>
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
