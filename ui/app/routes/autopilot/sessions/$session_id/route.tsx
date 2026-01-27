import type { Route } from "./+types/route";
import {
  Suspense,
  use,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
} from "react-router";
import { Loader2 } from "lucide-react";
import { PageHeader, Breadcrumbs } from "~/components/layout/PageLayout";
import EventStream, {
  type OptimisticMessage,
} from "~/components/autopilot/EventStream";
import { PendingToolCallCard } from "~/components/autopilot/PendingToolCallCard";
import { ChatInput } from "~/components/autopilot/ChatInput";
import { logger } from "~/utils/logger";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { useAutopilotEventStream } from "~/hooks/useAutopilotEventStream";
import type { AutopilotStatus, Event } from "~/types/tensorzero";
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

export type EventsData = {
  events: Event[];
  hasMoreEvents: boolean;
  pendingToolCalls: Event[];
  status: AutopilotStatus;
};

export async function loader({ params }: Route.LoaderArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    throw data("Session ID is required", { status: 400 });
  }

  // Special case: "new" session - return synchronously (no data to fetch)
  if (sessionId === "new") {
    return {
      sessionId: "new",
      eventsData: {
        events: [] as Event[],
        hasMoreEvents: false,
        pendingToolCalls: [] as Event[],
        status: { status: "idle" } as AutopilotStatus,
      },
      isNewSession: true,
    };
  }

  const client = getAutopilotClient();

  // Return promise WITHOUT awaiting - enables streaming/skeleton loading
  const eventsDataPromise = client
    .listAutopilotEvents(sessionId, {
      limit: EVENTS_PER_PAGE + 1,
    })
    .then((response) => {
      const hasMoreEvents = response.events.length > EVENTS_PER_PAGE;
      const events = response.events
        .sort(
          (a, b) =>
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
        )
        .slice(hasMoreEvents ? 1 : 0);
      // Sort pending tool calls by creation time (oldest first for queue)
      const pendingToolCalls = response.pending_tool_calls.sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
      return {
        events,
        hasMoreEvents,
        pendingToolCalls,
        status: response.status,
      };
    });

  return {
    sessionId,
    eventsData: eventsDataPromise,
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

// Skeleton shown while events are loading
function EventStreamSkeleton() {
  return (
    <div className="border-border mt-4 flex min-h-0 flex-1 items-center justify-center overflow-y-auto rounded-lg border p-4">
      <Loader2 className="text-muted-foreground h-8 w-8 animate-spin" />
    </div>
  );
}

// Main content component that resolves the promise and renders the event stream with SSE
function EventStreamContent({
  sessionId,
  eventsData,
  isNewSession,
  optimisticMessages,
  onOptimisticMessagesChange,
  scrollContainerRef,
  onLoaded,
  onStatusChange,
}: {
  sessionId: string;
  eventsData: EventsData | Promise<EventsData>;
  isNewSession: boolean;
  optimisticMessages: OptimisticMessage[];
  onOptimisticMessagesChange: (messages: OptimisticMessage[]) => void;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onLoaded: () => void;
  onStatusChange: (status: AutopilotStatus) => void;
}) {
  // Resolve promise (or use direct data for new session)
  const {
    events: initialEvents,
    hasMoreEvents: initialHasMore,
    pendingToolCalls: initialPendingToolCalls,
    status: initialStatus,
  } = eventsData instanceof Promise ? use(eventsData) : eventsData;

  // Signal that loading is complete (this runs after promise resolves)
  useEffect(() => {
    onLoaded();
  }, [onLoaded]);

  // Now that we have resolved events, start SSE with the correct lastEventId
  const { events, pendingToolCalls, status, error, isRetrying, prependEvents } =
    useAutopilotEventStream({
      sessionId: isNewSession ? NIL_UUID : sessionId,
      initialEvents,
      initialPendingToolCalls,
      initialStatus,
      enabled: !isNewSession,
    });

  // Notify parent of status changes
  useEffect(() => {
    onStatusChange(status);
  }, [status, onStatusChange]);

  // State for pagination
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [hasReachedStart, setHasReachedStart] = useState(!initialHasMore);

  // State for tool call authorization loading
  const [authLoadingStates, setAuthLoadingStates] = useState<
    Map<string, "approving" | "rejecting">
  >(new Map());

  const { toast } = useToast();

  // Derive values for queue-based approval UI
  const pendingToolCallIds = useMemo(
    () => new Set(pendingToolCalls.map((e) => e.id)),
    [pendingToolCalls],
  );
  const oldestPendingToolCall = pendingToolCalls[0] ?? null;

  // Cooldown animation: triggers when the queue top changes due to SSE (not user action).
  // Covers both directions: new item jumping to top, or top item removed by external approval.
  // Does NOT trigger when queue was empty and first item arrives (no accidental click risk).
  const prevQueueTopRef = useRef<string | null>(null);
  const userActionRef = useRef(false);
  const [isInCooldown, setIsInCooldown] = useState(false);

  useEffect(() => {
    const currentTopId = oldestPendingToolCall?.id ?? null;
    const prevTopId = prevQueueTopRef.current;

    prevQueueTopRef.current = currentTopId;
    const wasUserAction = userActionRef.current;
    userActionRef.current = false;

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

  /*
   * SCROLL BEHAVIOR SPEC:
   * 1. Submit message → Scroll to bottom (after optimistic message appears)
   * 2. New SSE event → Scroll to bottom ONLY if within BOTTOM_THRESHOLD of bottom
   * 3. Page load → Scroll to bottom (once)
   * 4. Scroll up (infinite scroll) → Preserve scroll position (no layout shift)
   * 5. BOTTOM_THRESHOLD (100px) → Buffer to handle tool card appearance
   */
  const BOTTOM_THRESHOLD = 100;

  // Refs for scroll management
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);
  const hasInitiallyScrolledRef = useRef(false);

  // For preserving scroll position when loading older events
  const pendingScrollPreservation = useRef<{
    scrollHeight: number;
    scrollTop: number;
  } | null>(null);

  const checkIfAtBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    return (
      container.scrollHeight - container.scrollTop - container.clientHeight <
      BOTTOM_THRESHOLD
    );
  }, [scrollContainerRef]);

  const scrollToBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [scrollContainerRef]);

  const handleScroll = useCallback(() => {
    isAtBottomRef.current = checkIfAtBottom();
  }, [checkIfAtBottom]);

  // Handle scroll when events change
  useEffect(() => {
    // Still loading older events - wait for them to arrive before adjusting scroll
    if (isLoadingOlder) return;

    // Older events loaded - preserve scroll position
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

    // New events - only scroll if user was at bottom
    if (isAtBottomRef.current) {
      scrollToBottom();
    }
  }, [events, isLoadingOlder, scrollToBottom, scrollContainerRef]);

  // Initial scroll to bottom on page load (once)
  useEffect(() => {
    if (!hasInitiallyScrolledRef.current && scrollContainerRef.current) {
      scrollToBottom();
      hasInitiallyScrolledRef.current = true;
    }
  }, [scrollToBottom, scrollContainerRef]);

  // Load older events
  const loadOlderEvents = useCallback(async () => {
    if (isLoadingOlder || hasReachedStart || events.length === 0) return;

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
        logger.debug(
          `API returned ${response.status} when fetching older events, treating as session start`,
        );
        setHasReachedStart(true);
        return;
      }

      const responseData = (await response.json()) as { events: Event[] };

      logger.debug(
        `Loaded ${responseData.events.length} older events (requested ${EVENTS_PER_PAGE + 1})`,
      );

      if (responseData.events.length <= EVENTS_PER_PAGE) {
        logger.debug("Reached session start");
        setHasReachedStart(true);
      }

      if (responseData.events.length === 0) {
        return;
      }

      const olderEvents = responseData.events
        .sort(
          (a, b) =>
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
        )
        .slice(responseData.events.length > EVENTS_PER_PAGE ? 1 : 0);

      prependEvents(olderEvents);
    } catch (err) {
      logger.error("Failed to load older events:", err);
      setHasReachedStart(true);
    } finally {
      setIsLoadingOlder(false);
    }
  }, [
    isLoadingOlder,
    hasReachedStart,
    events,
    sessionId,
    prependEvents,
    scrollContainerRef,
  ]);

  const loadOlderEventsDebounced = useMemo(
    () => debounce(loadOlderEvents, 100),
    [loadOlderEvents],
  );

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
        rootMargin: "300px 0px 0px 0px",
        threshold: 0.1,
      },
    );

    observer.observe(sentinel);

    return () => {
      observer.disconnect();
    };
  }, [
    isLoadingOlder,
    hasReachedStart,
    loadOlderEventsDebounced,
    scrollContainerRef,
  ]);

  // SSE delivers event → remove optimistic message when real event arrives
  useEffect(() => {
    if (optimisticMessages.length === 0) return;

    const confirmedEventIds = new Set(events.map((e) => e.id));

    const hasConfirmedMessages = optimisticMessages.some((msg) =>
      confirmedEventIds.has(msg.eventId),
    );

    if (hasConfirmedMessages) {
      onOptimisticMessagesChange(
        optimisticMessages.filter((msg) => !confirmedEventIds.has(msg.eventId)),
      );
    }
  }, [events, optimisticMessages, onOptimisticMessagesChange]);

  const confirmedEventIds = new Set(events.map((e) => e.id));
  const visibleOptimisticMessages = optimisticMessages.filter(
    (msg) => !confirmedEventIds.has(msg.eventId),
  );

  return (
    <>
      {error && isRetrying && (
        <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          Failed to fetch events. Retrying...
        </div>
      )}
      <div
        ref={(el) => {
          // Update parent's ref to point to the actual scrollable container
          if (scrollContainerRef) {
            (
              scrollContainerRef as React.MutableRefObject<HTMLDivElement | null>
            ).current = el;
          }
        }}
        onScroll={handleScroll}
        className="border-border mt-4 min-h-0 flex-1 overflow-y-auto rounded-lg border p-4"
      >
        <EventStream
          events={events}
          isLoadingOlder={isLoadingOlder}
          hasReachedStart={isNewSession ? false : hasReachedStart}
          topSentinelRef={topSentinelRef}
          pendingToolCallIds={pendingToolCallIds}
          optimisticMessages={visibleOptimisticMessages}
          status={isNewSession ? undefined : status}
        />
      </div>

      {/* Pinned approval card - outside scroll container */}
      {oldestPendingToolCall && (
        <div className="mt-4">
          <PendingToolCallCard
            key={oldestPendingToolCall.id}
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
    </>
  );
}

export default function AutopilotSessionEventsPage({
  loaderData,
}: Route.ComponentProps) {
  const { sessionId, eventsData, isNewSession } = loaderData;
  const navigate = useNavigate();
  const { toast } = useToast();

  // Lift optimistic messages state to parent so ChatInput can work outside Suspense
  const [optimisticMessages, setOptimisticMessages] = useState<
    OptimisticMessage[]
  >([]);

  // Clear optimistic messages when session changes to prevent cross-session leakage
  useEffect(() => {
    setOptimisticMessages([]);
  }, [sessionId]);

  // Track if events are still loading (for disabling chat input)
  // New sessions have direct data (not a promise), so they're not loading
  const [isEventsLoading, setIsEventsLoading] = useState(
    !isNewSession && eventsData instanceof Promise,
  );

  // Reset loading state when session changes (useState initial value only applies on first mount)
  useEffect(() => {
    setIsEventsLoading(!isNewSession && eventsData instanceof Promise);
  }, [sessionId, isNewSession, eventsData]);

  // Track autopilot status for disabling submit
  const [autopilotStatus, setAutopilotStatus] = useState<AutopilotStatus>({
    status: "idle",
  });

  // Reset status when session changes
  useEffect(() => {
    setAutopilotStatus({ status: "idle" });
  }, [sessionId]);

  const handleStatusChange = useCallback((status: AutopilotStatus) => {
    setAutopilotStatus(status);
  }, []);

  // Disable submit unless status is idle or failed
  const submitDisabled =
    autopilotStatus.status !== "idle" && autopilotStatus.status !== "failed";

  // Ref for scroll container - shared between parent and EventStreamContent
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  const handleNavigateToSession = useCallback(
    (newSessionId: string) => {
      navigate(`/autopilot/sessions/${newSessionId}`);
    },
    [navigate],
  );

  // Optimistic message handler - works without needing events resolved
  const handleMessageSent = useCallback(
    (response: { event_id: string; session_id: string }, text: string) => {
      setOptimisticMessages((prev) => [
        ...prev,
        {
          tempId: crypto.randomUUID(),
          eventId: response.event_id,
          text,
          status: "sending",
        },
      ]);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const container = scrollContainerRef.current;
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        });
      });

      if (isNewSession) {
        handleNavigateToSession(response.session_id);
      }
    },
    [isNewSession, handleNavigateToSession],
  );

  const handleMessageFailed = useCallback(
    (err: Error) => {
      toast.error({
        title: isNewSession
          ? "Failed to create session"
          : "Failed to send message",
        description: err.message,
      });
    },
    [toast, isNewSession],
  );

  return (
    <div className="container mx-auto flex h-full flex-col px-8 py-8">
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={
              isNewSession
                ? [{ label: "Autopilot", href: "/autopilot/sessions" }]
                : [
                    { label: "Autopilot", href: "/autopilot/sessions" },
                    { label: sessionId, isIdentifier: true },
                  ]
            }
          />
        }
        name={isNewSession ? "New Session" : undefined}
      />

      <Suspense fallback={<EventStreamSkeleton />}>
        <EventStreamContentWrapper
          key={sessionId}
          sessionId={sessionId}
          eventsData={eventsData}
          isNewSession={isNewSession}
          optimisticMessages={optimisticMessages}
          onOptimisticMessagesChange={setOptimisticMessages}
          scrollContainerRef={scrollContainerRef}
          onLoaded={() => setIsEventsLoading(false)}
          onStatusChange={handleStatusChange}
        />
      </Suspense>

      {/* Chat input - always visible outside Suspense, disabled while loading */}
      <ChatInput
        sessionId={isNewSession ? NIL_UUID : sessionId}
        onMessageSent={handleMessageSent}
        onMessageFailed={handleMessageFailed}
        className="mt-4"
        isNewSession={isNewSession}
        disabled={isEventsLoading}
        submitDisabled={submitDisabled}
      />
    </div>
  );
}

// Wrapper that passes the scroll container ref back to parent
function EventStreamContentWrapper({
  sessionId,
  eventsData,
  isNewSession,
  optimisticMessages,
  onOptimisticMessagesChange,
  scrollContainerRef,
  onLoaded,
  onStatusChange,
}: {
  sessionId: string;
  eventsData: EventsData | Promise<EventsData>;
  isNewSession: boolean;
  optimisticMessages: OptimisticMessage[];
  onOptimisticMessagesChange: (messages: OptimisticMessage[]) => void;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onLoaded: () => void;
  onStatusChange: (status: AutopilotStatus) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <EventStreamContent
        sessionId={sessionId}
        eventsData={eventsData}
        isNewSession={isNewSession}
        optimisticMessages={optimisticMessages}
        onOptimisticMessagesChange={onOptimisticMessagesChange}
        scrollContainerRef={scrollContainerRef}
        onLoaded={onLoaded}
        onStatusChange={onStatusChange}
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
