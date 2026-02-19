import { useCallback, useEffect, useSyncExternalStore } from "react";
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

// ── Module-level external store ──────────────────────────────────────────────
// Shared across all useEntitySheet() instances so that when UuidLink calls
// openInferenceSheet, EntitySheet's instance sees the update immediately.

let sheetState: EntitySheetState = null;
const listeners = new Set<() => void>();

function getSnapshot(): EntitySheetState {
  return sheetState;
}

function getServerSnapshot(): EntitySheetState {
  return null;
}

function emitChange(next: EntitySheetState) {
  sheetState = next;
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// ── Hook ─────────────────────────────────────────────────────────────────────

/**
 * Manages entity sheet state via URL search params using the History API.
 *
 * Uses window.history.pushState/replaceState instead of React Router's
 * setSearchParams to avoid triggering loader revalidation, which would
 * reset streamed/paginated state (e.g., autopilot event streams).
 *
 * State is shared across all hook instances via useSyncExternalStore so
 * that opening the sheet from UuidLink is visible to EntitySheet.
 */
export function useEntitySheet() {
  const location = useLocation();
  const currentState = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );

  // Initialize from URL on first mount (handles deep links / page refresh)
  useEffect(() => {
    const urlState = parseSheetStateFromUrl();
    if (urlState) {
      emitChange(urlState);
    }
  }, []);

  // Sync state on browser back/forward
  useEffect(() => {
    const handler = () => emitChange(parseSheetStateFromUrl());
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);

  // Sync state on React Router navigations (which don't fire popstate).
  // When React Router navigates, it replaces the URL, clearing our sheet params.
  useEffect(() => {
    emitChange(parseSheetStateFromUrl());
  }, [location.pathname, location.search]);

  const openInferenceSheet = useCallback((id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SHEET_PARAM, "inference");
    url.searchParams.set(SHEET_ID_PARAM, id);
    window.history.pushState(null, "", url.toString());
    emitChange({ type: "inference", id });
  }, []);

  // Use replaceState (not pushState) so that closing via X doesn't create
  // a new history entry, which would cause back-button cycling between
  // open and closed states.
  const closeSheet = useCallback(() => {
    const url = new URL(window.location.href);
    url.searchParams.delete(SHEET_PARAM);
    url.searchParams.delete(SHEET_ID_PARAM);
    window.history.replaceState(null, "", url.toString());
    emitChange(null);
  }, []);

  return { sheetState: currentState, openInferenceSheet, closeSheet };
}
