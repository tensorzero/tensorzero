import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import debounce from "lodash-es/debounce";
import { logger } from "~/utils/logger";

type ScrollPreservation = {
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

/**
 * Hook for infinite scroll pagination that loads older items when scrolling up.
 *
 * Features:
 * - Race condition prevention via synchronous ref guards
 * - Stable debounced function that doesn't recreate on dependency changes
 * - IntersectionObserver that doesn't reconnect on state changes
 * - Automatic scroll position preservation when new items are prepended
 * - Automatic retry with exponential backoff on transient errors
 * - Manual retry function for user-initiated recovery
 * - Cleanup on unmount
 *
 * @example
 * ```tsx
 * const { isLoadingOlder, hasReachedStart, loadError, topSentinelRef, retry } = useInfiniteScrollUp({
 *   items: events,
 *   initialHasMore: true,
 *   fetchOlder: async (oldest) => {
 *     const response = await fetch(`/api/events?before=${oldest.id}`);
 *     const data = await response.json();
 *     return { items: data.events, hasMore: data.hasMore };
 *   },
 *   prependItems: (newEvents) => setEvents(prev => [...newEvents, ...prev]),
 *   scrollContainerRef,
 * });
 *
 * return (
 *   <div ref={scrollContainerRef}>
 *     {loadError ? (
 *       <ErrorNotice message={loadError} onRetry={retry} />
 *     ) : hasReachedStart ? (
 *       <SessionStart />
 *     ) : (
 *       <div ref={topSentinelRef} />
 *     )}
 *     {isLoadingOlder && <Spinner />}
 *     {events.map(e => <Event key={e.id} event={e} />)}
 *   </div>
 * );
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
  // State for UI rendering
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [hasReachedStart, setHasReachedStart] = useState(!initialHasMore);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Refs for synchronous guards - prevent race conditions when called
  // multiple times before React re-renders
  const isLoadingOlderRef = useRef(false);
  const hasReachedStartRef = useRef(!initialHasMore);
  const hasErrorRef = useRef(false);

  // Retry tracking
  const retryCountRef = useRef(0);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sentinel element ref
  const topSentinelRef = useRef<HTMLDivElement | null>(null);

  // For preserving scroll position when loading older items
  const pendingScrollPreservation = useRef<ScrollPreservation | null>(null);

  // Clear any pending retry timeout on unmount
  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, []);

  // Load older items
  const loadOlderItems = useCallback(
    async (isRetry = false) => {
      // Use refs for synchronous guard - prevents race conditions when called
      // multiple times before React re-renders
      if (
        isLoadingOlderRef.current ||
        hasReachedStartRef.current ||
        items.length === 0
      ) {
        return;
      }

      // Don't auto-trigger if there's an error (user must click retry)
      // But allow if this is an explicit retry
      if (hasErrorRef.current && !isRetry) {
        return;
      }

      // Set ref synchronously BEFORE any async work to prevent duplicate calls
      isLoadingOlderRef.current = true;

      // Clear error state when attempting to load
      if (hasErrorRef.current) {
        hasErrorRef.current = false;
        setLoadError(null);
      }

      const container = scrollContainerRef.current;
      if (container) {
        pendingScrollPreservation.current = {
          scrollHeight: container.scrollHeight,
          scrollTop: container.scrollTop,
        };
      }

      setIsLoadingOlder(true);

      try {
        const oldestItem = items[0];
        const result = await fetchOlder(oldestItem);

        // Success - reset retry count
        retryCountRef.current = 0;

        if (!result.hasMore) {
          hasReachedStartRef.current = true;
          setHasReachedStart(true);
        }

        if (result.items.length > 0) {
          prependItems(result.items);
        }
      } catch (err) {
        logger.error("Failed to load older items:", err);

        // Check if we should auto-retry
        if (retryCountRef.current < maxRetries) {
          retryCountRef.current += 1;
          const delay =
            retryBaseDelayMs * Math.pow(2, retryCountRef.current - 1);
          logger.debug(
            `Auto-retrying pagination (attempt ${retryCountRef.current}/${maxRetries}) in ${delay}ms`,
          );

          // Schedule retry
          retryTimeoutRef.current = setTimeout(() => {
            retryTimeoutRef.current = null; // Clear before calling so finally knows we're not waiting
            isLoadingOlderRef.current = false;
            loadOlderItems(true);
          }, delay);

          // Keep loading state true during retry wait
          return;
        }

        // Max retries exceeded - show error to user
        hasErrorRef.current = true;
        setLoadError("Failed to load older messages");
        // Clear scroll preservation since we failed
        pendingScrollPreservation.current = null;
      } finally {
        // Only clear loading state if we're not waiting for a scheduled retry
        if (retryTimeoutRef.current === null) {
          isLoadingOlderRef.current = false;
          setIsLoadingOlder(false);
        }
      }
    },
    [
      items,
      fetchOlder,
      prependItems,
      scrollContainerRef,
      maxRetries,
      retryBaseDelayMs,
    ],
  );

  // Manual retry function for user-initiated recovery
  const retry = useCallback(() => {
    // Cancel any pending auto-retry
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }

    // Reset retry count for fresh attempt
    retryCountRef.current = 0;
    isLoadingOlderRef.current = false;

    // Trigger load with isRetry=true to bypass error guard
    loadOlderItems(true);
  }, [loadOlderItems]);

  // Keep a ref to the latest loadOlderItems so debounced function always calls current version
  const loadOlderItemsRef = useRef(loadOlderItems);
  useEffect(() => {
    loadOlderItemsRef.current = loadOlderItems;
  }, [loadOlderItems]);

  // Create a stable debounced function that:
  // 1. Always calls the latest loadOlderItems via ref (avoids stale closures)
  // 2. Never recreates (empty deps), so pending calls aren't lost
  // 3. Is cancelled on unmount
  const loadOlderItemsDebounced = useMemo(
    () =>
      debounce(() => {
        // Check refs inside callback - observer doesn't need to reconnect when these change
        if (
          isLoadingOlderRef.current ||
          hasReachedStartRef.current ||
          hasErrorRef.current
        ) {
          return;
        }
        loadOlderItemsRef.current();
      }, debounceMs),
    [debounceMs],
  );

  // Cancel pending debounced calls on unmount
  useEffect(() => {
    return () => {
      loadOlderItemsDebounced.cancel();
    };
  }, [loadOlderItemsDebounced]);

  // IntersectionObserver for infinite scroll
  // Uses refs for state checks to avoid reconnecting observer on every state change
  useEffect(() => {
    const sentinel = topSentinelRef.current;
    const container = scrollContainerRef.current;

    if (!sentinel || !container) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting) {
          loadOlderItemsDebounced();
        }
      },
      {
        root: container,
        rootMargin,
        threshold: 0.1,
      },
    );

    observer.observe(sentinel);

    return () => {
      observer.disconnect();
    };
  }, [loadOlderItemsDebounced, scrollContainerRef, rootMargin]);

  // Preserve scroll position when older items are loaded
  useEffect(() => {
    // Still loading - wait for items to arrive before adjusting scroll
    if (isLoadingOlder) {
      return;
    }

    // Older items loaded - preserve scroll position
    if (pendingScrollPreservation.current) {
      const container = scrollContainerRef.current;
      if (container) {
        const { scrollHeight: prevHeight, scrollTop: prevScrollTop } =
          pendingScrollPreservation.current;
        const heightDiff = container.scrollHeight - prevHeight;
        container.scrollTop = prevScrollTop + heightDiff;
      }
      pendingScrollPreservation.current = null;
    }
  }, [items, isLoadingOlder, scrollContainerRef]);

  return {
    isLoadingOlder,
    hasReachedStart,
    loadError,
    topSentinelRef,
    retry,
  };
}
