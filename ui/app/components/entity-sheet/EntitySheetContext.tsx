import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useLocation } from "react-router";

type EntitySheetState = { type: "inference"; id: string } | null;

interface EntitySheetContextValue {
  sheetState: EntitySheetState;
  openInferenceSheet: (id: string) => void;
  closeSheet: () => void;
}

const EntitySheetContext = createContext<EntitySheetContextValue | null>(null);

export function EntitySheetProvider({ children }: { children: ReactNode }) {
  const [sheetState, setSheetState] = useState<EntitySheetState>(null);
  const location = useLocation();

  // Close sheet on navigation
  useEffect(() => {
    setSheetState(null);
  }, [location.pathname]);

  const openInferenceSheet = useCallback((id: string) => {
    setSheetState({ type: "inference", id });
  }, []);

  const closeSheet = useCallback(() => {
    setSheetState(null);
  }, []);

  return (
    <EntitySheetContext.Provider
      value={{ sheetState, openInferenceSheet, closeSheet }}
    >
      {children}
    </EntitySheetContext.Provider>
  );
}

export function useEntitySheet() {
  const context = useContext(EntitySheetContext);
  if (!context) {
    throw new Error("useEntitySheet must be used within EntitySheetProvider");
  }
  return context;
}
