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

type EntitySheetState =
  | { type: "inference"; id: string }
  | { type: "episode"; id: string }
  | { type: "model_inference"; id: string }
  | null;

function parseSheetStateFromUrl(): EntitySheetState {
  const params = new URLSearchParams(window.location.search);
  const type = params.get(SHEET_PARAM);
  const id = params.get(SHEET_ID_PARAM);
  if (type === "inference" && id) {
    return { type, id };
  }
  if (type === "episode" && id) {
    return { type, id };
  }
  if (type === "model_inference" && id) {
    return { type, id };
  }
  return null;
}

interface EntitySheetContextValue {
  sheetState: EntitySheetState;
  openInferenceSheet: (id: string) => void;
  openEpisodeSheet: (id: string) => void;
  openModelInferenceSheet: (id: string) => void;
  closeSheet: () => void;
}

const EntitySheetContext = createContext<EntitySheetContextValue | null>(null);

/**
 * Manages entity sheet state as URL search params (?sheet=inference&sheetId=…).
 *
 * URL-backed state serves three purposes:
 *  1. Deep links — shareable/bookmarkable URLs with a sheet already open
 *     (e.g., an autopilot session with a particular inference highlighted).
 *  2. Reversible progressive disclosure — each drill-down step (open sheet,
 *     navigate to full page) is a history entry, so the back button rewinds
 *     the same path with no surprises.
 *  3. Dev hot-reload — sheet state survives HMR.
 *
 * We use window.history.pushState/replaceState rather than React Router's
 * setSearchParams to avoid triggering loader revalidation, which would reset
 * streamed or paginated state (e.g., autopilot event streams).
 */
export function EntitySheetProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [sheetState, setSheetState] = useState<EntitySheetState>(null);

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

  const openEpisodeSheet = useCallback((id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SHEET_PARAM, "episode");
    url.searchParams.set(SHEET_ID_PARAM, id);
    window.history.pushState(null, "", url.toString());
    setSheetState({ type: "episode", id });
  }, []);

  const openModelInferenceSheet = useCallback((id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SHEET_PARAM, "model_inference");
    url.searchParams.set(SHEET_ID_PARAM, id);
    window.history.pushState(null, "", url.toString());
    setSheetState({ type: "model_inference", id });
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
      value={{
        sheetState,
        openInferenceSheet,
        openEpisodeSheet,
        openModelInferenceSheet,
        closeSheet,
      }}
    >
      {children}
    </EntitySheetContext.Provider>
  );
}

export function useEntitySheet() {
  const context = useContext(EntitySheetContext);
  if (!context) {
    throw new Error(
      "`useEntitySheet` must be used within `EntitySheetProvider`",
    );
  }
  return context;
}
