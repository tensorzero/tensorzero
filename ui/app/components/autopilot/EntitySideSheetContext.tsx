import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

type EntitySheetState = { type: "inference"; id: string } | null;

interface EntitySideSheetContextValue {
  sheetState: EntitySheetState;
  openInferenceSheet: (id: string) => void;
  closeSheet: () => void;
}

const EntitySideSheetContext =
  createContext<EntitySideSheetContextValue | null>(null);

export function EntitySideSheetProvider({ children }: { children: ReactNode }) {
  const [sheetState, setSheetState] = useState<EntitySheetState>(null);

  const openInferenceSheet = useCallback((id: string) => {
    setSheetState({ type: "inference", id });
  }, []);

  const closeSheet = useCallback(() => {
    setSheetState(null);
  }, []);

  return (
    <EntitySideSheetContext.Provider
      value={{ sheetState, openInferenceSheet, closeSheet }}
    >
      {children}
    </EntitySideSheetContext.Provider>
  );
}

export function useEntitySideSheet() {
  const context = useContext(EntitySideSheetContext);
  if (!context) {
    throw new Error(
      "useEntitySideSheet must be used within EntitySideSheetProvider",
    );
  }
  return context;
}
