import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useLocation } from "react-router";

const SHEET_PARAM = "sheet";
const SHEET_ID_PARAM = "sheetId";

type EntitySheetState = { type: "inference"; id: string } | null;

function parseSheetStateFromUrl(): EntitySheetState {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  const type = params.get(SHEET_PARAM);
  const id = params.get(SHEET_ID_PARAM);
  if (type === "inference" && id) {
    return { type, id };
  }
  return null;
}

interface EntitySheetContextValue {
  sheetState: EntitySheetState;
  openInferenceSheet: (id: string) => void;
  closeSheet: () => void;
}

const EntitySheetContext = createContext<EntitySheetContextValue | null>(null);

/**
 * Provides entity sheet state via URL search params using the History API.
 *
 * Uses window.history.pushState/replaceState instead of React Router's
 * setSearchParams to avoid triggering loader revalidation, which would
 * reset streamed/paginated state (e.g., autopilot event streams).
 */
export function EntitySheetProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [sheetState, setSheetState] = useState<EntitySheetState>(null);

  // Initialize from URL on first mount (handles deep links / page refresh)
  useEffect(() => {
    const urlState = parseSheetStateFromUrl();
    if (urlState) {
      setSheetState(urlState);
    }
  }, []);

  // Sync state on browser back/forward
  useEffect(() => {
    const handler = () => setSheetState(parseSheetStateFromUrl());
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);

  // Sync state on React Router navigations (which don't fire popstate).
  // When React Router navigates, it replaces the URL, clearing our sheet params.
  useEffect(() => {
    setSheetState(parseSheetStateFromUrl());
  }, [location.pathname, location.search]);

  const openInferenceSheet = useCallback((id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SHEET_PARAM, "inference");
    url.searchParams.set(SHEET_ID_PARAM, id);
    window.history.pushState(null, "", url.toString());
    setSheetState({ type: "inference", id });
  }, []);

  // Use replaceState (not pushState) so that closing via X doesn't create
  // a new history entry, which would cause back-button cycling between
  // open and closed states.
  const closeSheet = useCallback(() => {
    const url = new URL(window.location.href);
    url.searchParams.delete(SHEET_PARAM);
    url.searchParams.delete(SHEET_ID_PARAM);
    window.history.replaceState(null, "", url.toString());
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
