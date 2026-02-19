import { useCallback, useEffect, useState } from "react";
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

/**
 * Manages entity sheet state via URL search params using direct history API.
 *
 * Uses window.history.pushState instead of React Router's setSearchParams
 * to avoid triggering loader revalidation, which would reset streamed/paginated
 * state (e.g., autopilot event streams).
 */
export function useEntitySheet() {
  const location = useLocation();
  const [sheetState, setSheetState] = useState<EntitySheetState>(
    parseSheetStateFromUrl,
  );

  // Sync state on browser back/forward (popstate doesn't fire for pushState)
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

  const closeSheet = useCallback(() => {
    const url = new URL(window.location.href);
    url.searchParams.delete(SHEET_PARAM);
    url.searchParams.delete(SHEET_ID_PARAM);
    window.history.pushState(null, "", url.toString());
    setSheetState(null);
  }, []);

  return { sheetState, openInferenceSheet, closeSheet };
}
