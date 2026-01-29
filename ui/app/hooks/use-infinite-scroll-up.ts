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
};

export type UseInfiniteScrollUpResult = {
  /** Whether older items are currently being loaded */
  isLoadingOlder: boolean;
  /** Whether we've reached the start (no more older items) */
  hasReachedStart: boolean;
  /** Ref to attach to the sentinel element at the top */
  topSentinelRef: React.RefObject<HTMLDivElement | null>;
};

/**
 * Hook for infinite scroll pagination that loads older items when scrolling up.
 *
 * Features:
 * - Race condition prevention via synchronous ref guards
 * - Stable debounced function that doesn't recreate on dependency changes
 * - IntersectionObserver that doesn't reconnect on state changes
 * - Automatic scroll position preservation when new items are prepended
 * - Cleanup on unmount
 *
 * @example
 * ```tsx
 * const { isLoadingOlder, hasReachedStart, topSentinelRef } = useInfiniteScrollUp({
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
 *     <div ref={topSentinelRef} /> {/* Sentinel at top *\/}
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
}: UseInfiniteScrollUpOptions<T>): UseInfiniteScrollUpResult {
  // State for UI rendering
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [hasReachedStart, setHasReachedStart] = useState(!initialHasMore);

  // Refs for synchronous guards - prevent race conditions when called
  // multiple times before React re-renders
  const isLoadingOlderRef = useRef(false);
  const hasReachedStartRef = useRef(!initialHasMore);

  // Sentinel element ref
  const topSentinelRef = useRef<HTMLDivElement | null>(null);

  // For preserving scroll position when loading older items
  const pendingScrollPreservation = useRef<ScrollPreservation | null>(null);

  // Load older items
  const loadOlderItems = useCallback(async () => {
    // Use refs for synchronous guard - prevents race conditions when called
    // multiple times before React re-renders
    if (
      isLoadingOlderRef.current ||
      hasReachedStartRef.current ||
      items.length === 0
    ) {
      return;
    }

    // Set ref synchronously BEFORE any async work to prevent duplicate calls
    isLoadingOlderRef.current = true;

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

      if (!result.hasMore) {
        hasReachedStartRef.current = true;
        setHasReachedStart(true);
      }

      if (result.items.length > 0) {
        prependItems(result.items);
      }
    } catch (err) {
      logger.error("Failed to load older items:", err);
      hasReachedStartRef.current = true;
      setHasReachedStart(true);
    } finally {
      isLoadingOlderRef.current = false;
      setIsLoadingOlder(false);
    }
  }, [items, fetchOlder, prependItems, scrollContainerRef]);

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
        if (isLoadingOlderRef.current || hasReachedStartRef.current) {
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
    topSentinelRef,
  };
}
