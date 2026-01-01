import type { Route } from "./+types/route";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { data, isRouteErrorResponse, type RouteHandle } from "react-router";
import { PageHeader } from "~/components/layout/PageLayout";
import EventStream from "~/components/autopilot/EventStream";
import { logger } from "~/utils/logger";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { useAutopilotEventStream } from "~/hooks/useAutopilotEventStream";
import type { Event } from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.session_id!, isIdentifier: true }],
};

const EVENTS_PER_PAGE = 20;

export async function loader({ params }: Route.LoaderArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    throw data("Session ID is required", { status: 400 });
  }

  const client = getAutopilotClient();
  // Fetch limit + 1 to detect if there are more events
  const response = await client.listAutopilotEvents(sessionId, {
    limit: EVENTS_PER_PAGE + 1,
  });

  // Check if there are more events than the page size
  const hasMoreEvents = response.events.length > EVENTS_PER_PAGE;

  // Sort events by created_at ascending and slice to page size
  const events = response.events
    .sort(
      (a, b) =>
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    )
    .slice(hasMoreEvents ? 1 : 0); // If hasMore, remove oldest event (first after sorting)

  return {
    sessionId,
    events,
    hasMoreEvents,
  };
}

// Simple debounce helper
function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delay: number,
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
      timeoutId = null;
    }, delay);
  };
}

export default function AutopilotSessionEventsPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    sessionId,
    events: initialEvents,
    hasMoreEvents: initialHasMore,
  } = loaderData;

  const { events, error, isRetrying, prependEvents } = useAutopilotEventStream({
    sessionId,
    initialEvents,
    // enabled: false, // Disabled while implementing pagination
  });

  // State for pagination
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [hasReachedStart, setHasReachedStart] = useState(!initialHasMore);

  // Refs
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  // Track if older events were just loaded (for scroll preservation)
  const pendingScrollPreservation = useRef<{
    scrollHeight: number;
    scrollTop: number;
  } | null>(null);

  // Stick-to-bottom behavior
  const checkIfAtBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    // Consider "at bottom" if within 50px of the bottom
    const threshold = 50;
    return (
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold
    );
  }, []);

  const scrollToBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, []);

  // Update isAtBottom when user scrolls
  const handleScroll = useCallback(() => {
    isAtBottomRef.current = checkIfAtBottom();
  }, [checkIfAtBottom]);

  // Preserve scroll position when content changes at the top (skeleton or events)
  useEffect(() => {
    // If we have pending scroll preservation, apply it first
    if (pendingScrollPreservation.current) {
      const container = scrollContainerRef.current;
      if (container) {
        const { scrollHeight: prevHeight, scrollTop: prevScrollTop } =
          pendingScrollPreservation.current;
        const heightDiff = container.scrollHeight - prevHeight;
        container.scrollTop = prevScrollTop + heightDiff;
      }
      pendingScrollPreservation.current = null;
      return;
    }

    // Otherwise, stick to bottom if user was at bottom
    if (isAtBottomRef.current) {
      scrollToBottom();
    }
  }, [events, isLoadingOlder, scrollToBottom]);

  // Scroll to bottom on initial mount
  useEffect(() => {
    scrollToBottom();
  }, [scrollToBottom]);

  // Load older events
  const loadOlderEvents = useCallback(async () => {
    if (isLoadingOlder || hasReachedStart || events.length === 0) return;

    // Store scroll position before loading
    const container = scrollContainerRef.current;
    if (container) {
      pendingScrollPreservation.current = {
        scrollHeight: container.scrollHeight,
        scrollTop: container.scrollTop,
      };
    }

    setIsLoadingOlder(true);

    try {
      const oldestEvent = events[0];
      const response = await fetch(
        `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events?limit=${EVENTS_PER_PAGE + 1}&before=${oldestEvent.id}`,
      );

      if (!response.ok) {
        // API might return 500 when there are no older events
        // Treat this as reaching the start of the session
        logger.debug(
          `API returned ${response.status} when fetching older events, treating as session start`,
        );
        setHasReachedStart(true);
        return;
      }

      const data = (await response.json()) as { events: Event[] };

      logger.debug(
        `Loaded ${data.events.length} older events (requested ${EVENTS_PER_PAGE + 1})`,
      );

      // Check if we've reached the start (fewer events than requested)
      if (data.events.length <= EVENTS_PER_PAGE) {
        logger.debug("Reached session start");
        setHasReachedStart(true);
      }

      // If no events returned, we're at the start
      if (data.events.length === 0) {
        return;
      }

      // Sort and take only the page size
      const olderEvents = data.events
        .sort(
          (a, b) =>
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
        )
        .slice(data.events.length > EVENTS_PER_PAGE ? 1 : 0);

      prependEvents(olderEvents);
    } catch (err) {
      // Network errors or other issues - treat as session start to avoid infinite retries
      logger.error("Failed to load older events:", err);
      setHasReachedStart(true);
    } finally {
      setIsLoadingOlder(false);
    }
  }, [isLoadingOlder, hasReachedStart, events, sessionId, prependEvents]);

  // Debounced version of loadOlderEvents
  const loadOlderEventsDebounced = useMemo(
    () => debounce(loadOlderEvents, 100),
    [loadOlderEvents],
  );

  // Intersection Observer for loading older events
  useEffect(() => {
    const sentinel = topSentinelRef.current;
    const container = scrollContainerRef.current;

    if (!sentinel || !container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting && !isLoadingOlder && !hasReachedStart) {
          loadOlderEventsDebounced();
        }
      },
      {
        root: container,
        // Start loading 300px before reaching the top
        rootMargin: "300px 0px 0px 0px",
        threshold: 0.1,
      },
    );

    observer.observe(sentinel);

    return () => {
      observer.disconnect();
    };
  }, [isLoadingOlder, hasReachedStart, loadOlderEventsDebounced]);

  return (
    <div className="container mx-auto flex h-full flex-col px-8 pt-16 pb-8">
      <PageHeader label="Autopilot Session" name={sessionId} />
      {error && isRetrying && (
        <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          Failed to fetch events. Retrying...
        </div>
      )}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="border-border mt-4 min-h-0 flex-1 overflow-y-auto rounded-lg border p-4"
      >
        <EventStream
          events={events}
          isLoadingOlder={isLoadingOlder}
          hasReachedStart={hasReachedStart}
          topSentinelRef={topSentinelRef}
        />
      </div>
    </div>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
