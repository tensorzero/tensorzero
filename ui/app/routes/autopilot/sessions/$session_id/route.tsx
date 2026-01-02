import type { Route } from "./+types/route";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  data,
  isRouteErrorResponse,
  Link,
  useNavigate,
  type RouteHandle,
} from "react-router";
import { Plus } from "lucide-react";
import { PageHeader } from "~/components/layout/PageLayout";
import EventStream, {
  type OptimisticMessage,
} from "~/components/autopilot/EventStream";
import { PendingToolCallCard } from "~/components/autopilot/PendingToolCallCard";
import { ChatInput } from "~/components/autopilot/ChatInput";
import { logger } from "~/utils/logger";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { useAutopilotEventStream } from "~/hooks/useAutopilotEventStream";
import type { Event } from "~/types/tensorzero";
import { useToast } from "~/hooks/use-toast";

// Nil UUID for creating new sessions
const NIL_UUID = "00000000-0000-0000-0000-000000000000";

export const handle: RouteHandle = {
  crumb: (match) => [
    match.params.session_id === "new"
      ? { label: "New Session" }
      : { label: match.params.session_id!, isIdentifier: true },
  ],
};

const EVENTS_PER_PAGE = 20;

export async function loader({ params }: Route.LoaderArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    throw data("Session ID is required", { status: 400 });
  }

  // Special case: "new" session
  if (sessionId === "new") {
    return {
      sessionId: "new",
      events: [] as Event[],
      hasMoreEvents: false,
      isNewSession: true,
    };
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
    isNewSession: false,
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
    isNewSession,
  } = loaderData;

  const navigate = useNavigate();

  const { events, error, isRetrying, prependEvents } = useAutopilotEventStream({
    sessionId: isNewSession ? NIL_UUID : sessionId,
    initialEvents,
    enabled: !isNewSession, // Disable SSE for new sessions
  });

  // State for pagination
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [hasReachedStart, setHasReachedStart] = useState(!initialHasMore);

  // State for tool call authorization loading
  const [authLoadingStates, setAuthLoadingStates] = useState<
    Map<string, "approving" | "rejecting">
  >(new Map());

  const { toast } = useToast();

  // Optimistic messages: shown immediately when user sends, removed when SSE confirms.
  // See OptimisticMessage type in EventStream.tsx for detailed flow documentation.
  const [optimisticMessages, setOptimisticMessages] = useState<
    OptimisticMessage[]
  >([]);

  // Compute pending tool calls (tool_calls without matching authorization or result)
  const pendingToolCalls = useMemo(() => {
    const authorizedIds = new Set<string>();
    const resultIds = new Set<string>();

    for (const event of events) {
      if (event.payload.type === "tool_call_authorization") {
        authorizedIds.add(event.payload.tool_call_event_id);
      }
      if (event.payload.type === "tool_result") {
        resultIds.add(event.payload.tool_call_event_id);
      }
    }

    return events.filter(
      (e) =>
        e.payload.type === "tool_call" &&
        !authorizedIds.has(e.id) &&
        !resultIds.has(e.id),
    );
  }, [events]);

  // Derive values for queue-based approval UI
  const pendingToolCallIds = useMemo(
    () => new Set(pendingToolCalls.map((e) => e.id)),
    [pendingToolCalls],
  );
  const oldestPendingToolCall = pendingToolCalls[0] ?? null;

  // SSE change detection for queue top changes.
  // When the oldest pending tool call changes due to an external event (e.g., another user
  // or system approving/rejecting via SSE), we show a brief cooldown animation to prevent
  // accidental clicks on the newly displayed card. This gives users time to recognize
  // that the card content has changed before they can interact with it.
  const prevQueueTopRef = useRef<string | null>(null);
  const userActionRef = useRef(false);
  const [isInCooldown, setIsInCooldown] = useState(false);

  useEffect(() => {
    const currentTopId = oldestPendingToolCall?.id ?? null;
    const prevTopId = prevQueueTopRef.current;

    // Update refs
    prevQueueTopRef.current = currentTopId;
    const wasUserAction = userActionRef.current;
    userActionRef.current = false;

    // If top changed and it wasn't due to user action, trigger cooldown
    if (currentTopId !== prevTopId && prevTopId !== null && !wasUserAction) {
      setIsInCooldown(true);
      const timer = setTimeout(() => setIsInCooldown(false), 1000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [oldestPendingToolCall?.id]);

  // Handle tool call authorization
  const handleAuthorize = useCallback(
    async (eventId: string, approved: boolean) => {
      // Mark as user action to prevent cooldown when this authorization causes queue change
      userActionRef.current = true;

      setAuthLoadingStates((prev) =>
        new Map(prev).set(eventId, approved ? "approving" : "rejecting"),
      );

      try {
        const response = await fetch(
          `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events/authorize`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              tool_call_event_id: eventId,
              status: approved
                ? { type: "approved" }
                : {
                    type: "rejected",
                    reason: "The user rejected the tool call.",
                  },
            }),
          },
        );

        if (!response.ok) {
          throw new Error("Authorization failed");
        }
        // Card will disappear when authorization event arrives via SSE
      } catch (err) {
        logger.error("Failed to authorize tool call:", err);
        toast.error({
          title: "Authorization failed",
          description:
            "Failed to submit tool call authorization. Please try again.",
        });
      } finally {
        setAuthLoadingStates((prev) => {
          const next = new Map(prev);
          next.delete(eventId);
          return next;
        });
      }
    },
    [sessionId, toast],
  );

  // Optimistic message handler: create message only after POST completes (with eventId)
  // This eliminates the race condition where SSE could deliver before we have the ID
  const handleMessageSent = useCallback(
    (response: { event_id: string; session_id: string }, text: string) => {
      // Create optimistic message with eventId already set
      setOptimisticMessages((prev) => [
        ...prev,
        {
          tempId: crypto.randomUUID(),
          eventId: response.event_id,
          text,
          status: "sending",
        },
      ]);

      // Scroll to bottom after React re-renders with the new message
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const container = scrollContainerRef.current;
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        });
      });

      // For new sessions, navigate to the created session
      if (isNewSession) {
        navigate(`/autopilot/sessions/${response.session_id}`);
      }
    },
    [isNewSession, navigate],
  );

  const handleMessageFailed = useCallback(
    (error: Error) => {
      // No optimistic message exists on failure (we only create after POST succeeds)
      toast.error({
        title: isNewSession
          ? "Failed to create session"
          : "Failed to send message",
        description: error.message,
      });
    },
    [toast, isNewSession],
  );

  // SSE delivers event → remove optimistic message when real event arrives
  useEffect(() => {
    if (optimisticMessages.length === 0) return;

    const confirmedEventIds = new Set(events.map((e) => e.id));

    // Remove optimistic messages whose eventId matches a confirmed event
    const hasConfirmedMessages = optimisticMessages.some((msg) =>
      confirmedEventIds.has(msg.eventId),
    );

    if (hasConfirmedMessages) {
      setOptimisticMessages((prev) =>
        prev.filter((msg) => !confirmedEventIds.has(msg.eventId)),
      );
    }
  }, [events, optimisticMessages]);

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

  // Filter out optimistic messages that have a matching event in the stream
  // Since we only create optimistic messages after POST completes, eventId is always set
  const confirmedEventIds = new Set(events.map((e) => e.id));
  const visibleOptimisticMessages = optimisticMessages.filter(
    (msg) => !confirmedEventIds.has(msg.eventId),
  );

  return (
    <div className="container mx-auto flex h-full flex-col px-8 py-8">
      <PageHeader
        label="Autopilot Session"
        name={isNewSession ? "New Session" : sessionId}
        tag={
          !isNewSession ? (
            <Link
              to="/autopilot/sessions/new"
              className="text-fg-tertiary hover:text-fg-secondary ml-2 inline-flex items-center gap-1 text-sm font-medium transition-colors"
            >
              <Plus className="h-4 w-4" />
              New Session
            </Link>
          ) : undefined
        }
      />
      {error && isRetrying && (
        <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          Failed to fetch events. Retrying...
        </div>
      )}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="border-border mt-8 min-h-0 flex-1 overflow-y-auto rounded-lg border p-4"
      >
        <EventStream
          events={events}
          isLoadingOlder={isLoadingOlder}
          hasReachedStart={isNewSession ? false : hasReachedStart}
          topSentinelRef={topSentinelRef}
          pendingToolCallIds={pendingToolCallIds}
          optimisticMessages={visibleOptimisticMessages}
        />
      </div>

      {/* Pinned approval card - outside scroll container */}
      {oldestPendingToolCall && (
        <div className="mt-4">
          <PendingToolCallCard
            event={oldestPendingToolCall}
            isLoading={authLoadingStates.has(oldestPendingToolCall.id)}
            loadingAction={authLoadingStates.get(oldestPendingToolCall.id)}
            onAuthorize={(approved) =>
              handleAuthorize(oldestPendingToolCall.id, approved)
            }
            additionalCount={pendingToolCalls.length - 1}
            isInCooldown={isInCooldown}
          />
        </div>
      )}

      {/* Chat input */}
      <ChatInput
        sessionId={isNewSession ? NIL_UUID : sessionId}
        onMessageSent={handleMessageSent}
        onMessageFailed={handleMessageFailed}
        className="mt-4"
        isNewSession={isNewSession}
      />
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
