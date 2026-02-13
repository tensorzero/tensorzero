import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

interface InferenceSideSheetContextValue {
  inferenceId: string | null;
  openSheet: (inferenceId: string) => void;
  closeSheet: () => void;
}

const InferenceSideSheetContext =
  createContext<InferenceSideSheetContextValue | null>(null);

export function InferenceSideSheetProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [inferenceId, setInferenceId] = useState<string | null>(null);

  const openSheet = useCallback((id: string) => {
    setInferenceId(id);
  }, []);

  const closeSheet = useCallback(() => {
    setInferenceId(null);
  }, []);

  return (
    <InferenceSideSheetContext.Provider
      value={{ inferenceId, openSheet, closeSheet }}
    >
      {children}
    </InferenceSideSheetContext.Provider>
  );
}

export function useInferenceSideSheet() {
  const context = useContext(InferenceSideSheetContext);
  if (!context) {
    throw new Error(
      "useInferenceSideSheet must be used within InferenceSideSheetProvider",
    );
  }
  return context;
}
