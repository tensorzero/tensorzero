import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import debounce from "lodash-es/debounce";
import { logger } from "~/utils/logger";
import { useLatest } from "./use-latest";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type ScrollSnapshot = {
  scrollHeight: number;
  scrollTop: number;
};

export type UseInfiniteScrollUpOptions<T> = {
  /** The current list of items (used to get oldest item for pagination) */
  items: T[];
  /** Whether there are more items to load initially */
  initialHasMore: boolean;
  /** Async function to fetch older items. Returns items and whether there are more. */
  fetchOlder: (oldestItem: T) => Promise<{ items: T[]; hasMore: boolean }>;
  /** Function to prepend fetched items to the list */
  prependItems: (items: T[]) => void;
  /** Ref to the scroll container element */
  scrollContainerRef: React.RefObject<HTMLElement | null>;
  /** Debounce delay in ms (default: 100) */
  debounceMs?: number;
  /** Root margin for IntersectionObserver (default: "300px 0px 0px 0px") */
  rootMargin?: string;
  /** Max automatic retry attempts on error (default: 2) */
  maxRetries?: number;
  /** Base delay for exponential backoff in ms (default: 1000) */
  retryBaseDelayMs?: number;
};

export type UseInfiniteScrollUpResult = {
  /** Whether older items are currently being loaded */
  isLoadingOlder: boolean;
  /** Whether we've reached the start (no more older items) */
  hasReachedStart: boolean;
  /** Error message if loading failed (null if no error) */
  loadError: string | null;
  /** Ref to attach to the sentinel element at the top */
  topSentinelRef: React.RefObject<HTMLDivElement | null>;
  /** Manually retry loading after an error */
  retry: () => void;
};

// ─────────────────────────────────────────────────────────────────────────────
// State Machine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Load states:
 * - idle: Ready to load, no error
 * - loading: Fetch in progress
 * - waiting_retry: Failed, auto-retry scheduled
 * - error: Failed after max retries, waiting for manual retry
 * - reached_start: No more items to load
 */
type LoadStatus =
  | "idle"
  | "loading"
  | "waiting_retry"
  | "error"
  | "reached_start";

type LoadState = {
  status: LoadStatus;
  retryCount: number;
  errorMessage: string | null;
};

type LoadAction =
  | { type: "START_LOADING" }
  | { type: "FETCH_SUCCESS"; hasMore: boolean }
  | { type: "SCHEDULE_RETRY" }
  | { type: "MAX_RETRIES_EXCEEDED"; errorMessage: string }
  | { type: "RESET_FOR_RETRY" };

function loadReducer(state: LoadState, action: LoadAction): LoadState {
  switch (action.type) {
    case "START_LOADING":
      return {
        status: "loading",
        retryCount: state.retryCount,
        errorMessage: null,
      };

    case "FETCH_SUCCESS":
      return {
        status: action.hasMore ? "idle" : "reached_start",
        retryCount: 0,
        errorMessage: null,
      };

    case "SCHEDULE_RETRY":
      return {
        status: "waiting_retry",
        retryCount: state.retryCount + 1,
        errorMessage: null,
      };

    case "MAX_RETRIES_EXCEEDED":
      return {
        status: "error",
        retryCount: state.retryCount + 1,
        errorMessage: action.errorMessage,
      };

    case "RESET_FOR_RETRY":
      // Reset to idle so loadOlderItems can start fresh
      return { status: "idle", retryCount: 0, errorMessage: null };

    default:
      return state;
  }
}

function createInitialState(hasMore: boolean): LoadState {
  return {
    status: hasMore ? "idle" : "reached_start",
    retryCount: 0,
    errorMessage: null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

function canStartLoad(status: LoadStatus, itemCount: number): boolean {
  if (itemCount === 0) return false;
  if (status === "loading") return false;
  if (status === "reached_start") return false;
  // "idle", "error", and "waiting_retry" can start a load
  // (waiting_retry transitions to loading when the retry timer fires)
  return true;
}

function getRetryDelay(retryCount: number, baseDelayMs: number): number {
  return baseDelayMs * Math.pow(2, retryCount - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hook for infinite scroll pagination that loads older items when scrolling up.
 *
 * ## How It Works
 *
 * 1. **Trigger**: IntersectionObserver fires when sentinel near top becomes visible
 * 2. **Guard**: Check state synchronously via ref to prevent concurrent loads
 * 3. **Load**: Fetch older items, with auto-retry on transient errors
 * 4. **Restore**: Adjust scroll position so user stays at same visual spot
 *
 * ## State Machine
 *
 * ```
 * idle ──[START_LOADING]──► loading
 *                              │
 *              ┌───────────────┴───────────────┐
 *              ▼                               ▼
 *   [FETCH_SUCCESS]                     [on error]
 *         │                                   │
 *    ┌────┴────┐                    ┌─────────┴─────────┐
 *    ▼         ▼                    ▼                   ▼
 *  idle   reached_start      waiting_retry            error
 *                             (auto-retry)        (max retries)
 *                                   │                   │
 *                            [timer fires]       [RESET_FOR_RETRY]
 *                                   │                   │
 *                                   └─► [START_LOADING] ◄┘
 *                                              │
 *                                              ▼
 *                                           loading
 * ```
 */
export function useInfiniteScrollUp<T>({
  items,
  initialHasMore,
  fetchOlder,
  prependItems,
  scrollContainerRef,
  debounceMs = 100,
  rootMargin = "300px 0px 0px 0px",
  maxRetries = 2,
  retryBaseDelayMs = 1000,
}: UseInfiniteScrollUpOptions<T>): UseInfiniteScrollUpResult {
  // ─────────────────────────────────────────────────────────────────────────
  // State Machine
  // ─────────────────────────────────────────────────────────────────────────

  const [state, dispatch] = useReducer(loadReducer, initialHasMore, (hasMore) =>
    createInitialState(hasMore),
  );

  // Ref mirrors state for synchronous access in guards (before React re-renders)
  const stateRef = useRef(state);

  // Transition helper: updates ref synchronously, then dispatches to React
  const transition = useCallback((action: LoadAction) => {
    stateRef.current = loadReducer(stateRef.current, action);
    dispatch(action);
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Refs
  // ─────────────────────────────────────────────────────────────────────────

  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollSnapshotRef = useRef<ScrollSnapshot | null>(null);
  const topSentinelRef = useRef<HTMLDivElement | null>(null);

  // ─────────────────────────────────────────────────────────────────────────
  // Scroll Preservation
  // ─────────────────────────────────────────────────────────────────────────

  const captureScrollPosition = useCallback(() => {
    const container = scrollContainerRef.current;
    if (container) {
      scrollSnapshotRef.current = {
        scrollHeight: container.scrollHeight,
        scrollTop: container.scrollTop,
      };
    }
  }, [scrollContainerRef]);

  const restoreScrollPosition = useCallback(() => {
    const snapshot = scrollSnapshotRef.current;
    const container = scrollContainerRef.current;

    if (snapshot && container) {
      const heightDiff = container.scrollHeight - snapshot.scrollHeight;
      container.scrollTop = snapshot.scrollTop + heightDiff;
    }

    scrollSnapshotRef.current = null;
  }, [scrollContainerRef]);

  // ─────────────────────────────────────────────────────────────────────────
  // Cleanup
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Core Load Function
  // ─────────────────────────────────────────────────────────────────────────

  const loadOlderItems = useCallback(async () => {
    // Synchronous guard using ref (prevents race conditions before React re-renders)
    if (!canStartLoad(stateRef.current.status, items.length)) {
      return;
    }

    transition({ type: "START_LOADING" });
    captureScrollPosition();

    try {
      const result = await fetchOlder(items[0]);

      transition({ type: "FETCH_SUCCESS", hasMore: result.hasMore });

      if (result.items.length > 0) {
        prependItems(result.items);
      }
    } catch (err) {
      logger.error("Failed to load older items:", err);

      const currentRetryCount = stateRef.current.retryCount;
      const canRetry = currentRetryCount < maxRetries;

      if (canRetry) {
        transition({ type: "SCHEDULE_RETRY" });

        const delay = getRetryDelay(currentRetryCount + 1, retryBaseDelayMs);
        logger.debug(
          `Auto-retrying (${currentRetryCount + 1}/${maxRetries}) in ${delay}ms`,
        );

        retryTimeoutRef.current = setTimeout(() => {
          retryTimeoutRef.current = null;
          // loadOlderItems will transition from "waiting_retry" to "loading" via START_LOADING
          loadOlderItems();
        }, delay);
      } else {
        transition({
          type: "MAX_RETRIES_EXCEEDED",
          errorMessage: "Failed to load older messages",
        });
        scrollSnapshotRef.current = null;
      }
    }
  }, [
    items,
    fetchOlder,
    prependItems,
    maxRetries,
    retryBaseDelayMs,
    transition,
    captureScrollPosition,
  ]);

  // ─────────────────────────────────────────────────────────────────────────
  // Manual Retry
  // ─────────────────────────────────────────────────────────────────────────

  const retry = useCallback(() => {
    // Guard against rapid clicks - if already loading, ignore
    if (stateRef.current.status === "loading") {
      return;
    }

    // Cancel any pending auto-retry
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }

    // Reset state to allow a fresh load attempt
    transition({ type: "RESET_FOR_RETRY" });

    // Now loadOlderItems can proceed (status is "idle")
    loadOlderItems();
  }, [loadOlderItems, transition]);

  // ─────────────────────────────────────────────────────────────────────────
  // IntersectionObserver Trigger
  // ─────────────────────────────────────────────────────────────────────────

  // useLatest keeps a stable ref to the latest loadOlderItems without needing
  // to include it in dependency arrays (avoids observer reconnections)
  const loadOlderItemsRef = useLatest(loadOlderItems);

  const debouncedLoad = useMemo(
    () => debounce(() => loadOlderItemsRef.current(), debounceMs),
    [debounceMs, loadOlderItemsRef],
  );

  useEffect(() => {
    return () => debouncedLoad.cancel();
  }, [debouncedLoad]);

  useEffect(() => {
    const sentinel = topSentinelRef.current;
    const container = scrollContainerRef.current;

    if (!sentinel || !container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          debouncedLoad();
        }
      },
      { root: container, rootMargin, threshold: 0.1 },
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [debouncedLoad, scrollContainerRef, rootMargin]);

  // ─────────────────────────────────────────────────────────────────────────
  // Scroll Position Restoration
  // ─────────────────────────────────────────────────────────────────────────

  const isLoading =
    state.status === "loading" || state.status === "waiting_retry";

  useEffect(() => {
    if (!isLoading && scrollSnapshotRef.current) {
      restoreScrollPosition();
    }
  }, [items, isLoading, restoreScrollPosition]);

  // ─────────────────────────────────────────────────────────────────────────
  // Return Derived State
  // ─────────────────────────────────────────────────────────────────────────

  return {
    isLoadingOlder: isLoading,
    hasReachedStart: state.status === "reached_start",
    loadError: state.errorMessage,
    topSentinelRef,
    retry,
  };
}
