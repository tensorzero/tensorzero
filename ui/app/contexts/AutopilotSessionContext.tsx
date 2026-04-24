import { createContext, useContext, type ReactNode } from "react";
import { useLocalStorage } from "~/hooks/use-local-storage";

interface AutopilotSessionContextValue {
  autoApprove: boolean;
  setAutoApprove: (value: boolean) => void;
  /** @deprecated Use `autoApprove` instead */
  yoloMode: boolean;
  /** @deprecated Use `setAutoApprove` instead */
  setYoloMode: (value: boolean) => void;
}

const AutopilotSessionContext =
  createContext<AutopilotSessionContextValue | null>(null);

export function AutopilotSessionProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [autoApprove, setAutoApprove] = useLocalStorage<boolean>(
    "tensorzero-auto-approve",
    false,
  );

  return (
    <AutopilotSessionContext.Provider
      value={{
        autoApprove,
        setAutoApprove,
        yoloMode: autoApprove,
        setYoloMode: setAutoApprove,
      }}
    >
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
