import type { Route } from "./+types/route";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { v7 as uuid } from "uuid";
import {
  Await,
  data,
  useAsyncError,
  useFetcher,
  useNavigate,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
} from "react-router";
import { AlertCircle, AlertTriangle, Loader2 } from "lucide-react";
import { Breadcrumbs } from "~/components/layout/PageLayout";
import EventStream, {
  type OptimisticMessage,
} from "~/components/autopilot/EventStream";
import { PendingToolCallCard } from "~/components/autopilot/PendingToolCallCard";
import { YoloModeToggle } from "~/components/autopilot/YoloModeToggle";
import {
  AutopilotStatusBanner,
  AutopilotStatusBannerVariant,
} from "~/components/autopilot/AutopilotStatusBanner";
import { ChatInput } from "~/components/autopilot/ChatInput";
import { FadeDirection, FadeGradient } from "~/components/ui/FadeGradient";
import { fetchOlderAutopilotEvents } from "~/utils/autopilot/fetch-older-events";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { useAutopilotEventStream } from "~/hooks/useAutopilotEventStream";
import { useElementHeight } from "~/hooks/useElementHeight";
import { useInfiniteScrollUp } from "~/hooks/use-infinite-scroll-up";
import { useAutoApproval } from "~/hooks/use-auto-approval";
import { useManualAuthorization } from "~/hooks/use-manual-approval";
import type { AuthorizationLoadingAction } from "~/utils/autopilot/types";
import {
  AutopilotSessionProvider,
  useAutopilotSession,
} from "~/contexts/AutopilotSessionContext";
import type { AutopilotStatus, GatewayEvent } from "~/types/tensorzero";
import { useToast } from "~/hooks/use-toast";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { useFeatureFlags } from "~/context/feature-flags";

// Nil UUID for creating new sessions
const NIL_UUID = "00000000-0000-0000-0000-000000000000";

export const handle: RouteHandle = {
  crumb: (match) => [
    match.params.session_id === "new"
      ? { label: "New Session" }
      : { label: match.params.session_id!, isIdentifier: true },
  ],
};

/**
 * Prevent revalidation of this route when API actions are submitted.
 * The event stream already supplies fresh data, so revalidation can
 * overwrite SSE-delivered events with a short loader snapshot.
 */
export function shouldRevalidate({
  formAction,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  if (formAction?.startsWith("/api/autopilot/sessions/")) {
    return false;
  }
  return defaultShouldRevalidate;
}

const EVENTS_PER_PAGE = 25;

export type EventsData = {
  events: GatewayEvent[];
  hasMoreEvents: boolean;
  pendingToolCalls: GatewayEvent[];
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
        events: [] as GatewayEvent[],
        hasMoreEvents: false,
        pendingToolCalls: [] as GatewayEvent[],
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

// Skeleton shown while events are loading
function EventStreamSkeleton() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <Loader2 className="text-muted-foreground h-8 w-8 animate-spin" />
    </div>
  );
}

/**
 * Error state shown when initial event stream load fails.
 * Preserves the chat container layout so the page doesn't completely break.
 */
function EventStreamLoadError({ onError }: { onError: () => void }) {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load session events";

  // Notify parent that we're in error state (disables ChatInput)
  useEffect(() => {
    onError();
  }, [onError]);

  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <SectionErrorNotice
        icon={AlertCircle}
        title="Error loading session"
        description={message}
      />
    </div>
  );
}

// Main content component that renders the event stream with SSE
function EventStreamContent({
  sessionId,
  eventsData,
  isNewSession,
  optimisticMessages,
  onOptimisticMessagesChange,
  scrollContainerRef,
  onLoaded,
  onStatusChange,
  onPendingToolCallsChange,
  onErrorChange,
  onHasReachedStartChange,
  pendingToolCallIds,
}: {
  sessionId: string;
  eventsData: EventsData;
  isNewSession: boolean;
  optimisticMessages: OptimisticMessage[];
  onOptimisticMessagesChange: (messages: OptimisticMessage[]) => void;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onLoaded: () => void;
  onStatusChange: (status: AutopilotStatus) => void;
  onPendingToolCallsChange: (pendingToolCalls: GatewayEvent[]) => void;
  onErrorChange: (error: string | null, isRetrying: boolean) => void;
  onHasReachedStartChange: (hasReachedStart: boolean) => void;
  pendingToolCallIds: Set<string>;
}) {
  const {
    events: initialEvents,
    hasMoreEvents: initialHasMore,
    pendingToolCalls: initialPendingToolCalls,
    status: initialStatus,
  } = eventsData;

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

  // Notify parent of pending tool calls changes
  useEffect(() => {
    onPendingToolCallsChange(pendingToolCalls);
  }, [pendingToolCalls, onPendingToolCallsChange]);

  // Notify parent of error state changes
  useEffect(() => {
    onErrorChange(error, isRetrying);
  }, [error, isRetrying, onErrorChange]);

  // Fetch older events for infinite scroll pagination
  const fetchOlderEvents = useCallback(
    async (oldestEvent: GatewayEvent) => {
      return fetchOlderAutopilotEvents(sessionId, oldestEvent.id);
    },
    [sessionId],
  );

  // Infinite scroll pagination (loading older events when scrolling up)
  const {
    isLoadingOlder,
    hasReachedStart,
    loadError: paginationError,
    topSentinelRef,
    retry: retryPagination,
  } = useInfiniteScrollUp({
    items: events,
    initialHasMore: initialHasMore,
    fetchOlder: fetchOlderEvents,
    prependItems: prependEvents,
    scrollContainerRef,
  });

  // Notify parent when hasReachedStart changes (for top fade visibility)
  useEffect(() => {
    onHasReachedStartChange(hasReachedStart);
  }, [hasReachedStart, onHasReachedStartChange]);

  /*
   * SCROLL BEHAVIOR SPEC:
   * 1. Submit message → Scroll to bottom (after optimistic message appears)
   * 2. New SSE event → Scroll to bottom ONLY if within BOTTOM_THRESHOLD of bottom
   * 3. Page load → Scroll to bottom (once)
   * 4. Scroll up (infinite scroll) → Preserve scroll position (handled by useInfiniteScrollUp)
   * 5. BOTTOM_THRESHOLD (100px) → Buffer to handle tool card appearance
   */
  const BOTTOM_THRESHOLD = 100;

  // Refs for scroll-to-bottom management
  const isAtBottomRef = useRef(true);
  const hasInitiallyScrolledRef = useRef(false);

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

  // Scroll to bottom when new events arrive (SSE), but not when loading older events
  useEffect(() => {
    // Skip during pagination - useInfiniteScrollUp handles scroll preservation
    if (isLoadingOlder) return;

    // Scroll to bottom only if user was already at bottom
    if (isAtBottomRef.current) {
      scrollToBottom();
    }
  }, [events, isLoadingOlder, scrollToBottom]);

  // Listen to scroll events from the parent-provided scroll container
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [scrollContainerRef, handleScroll]);

  // Initial scroll to bottom on page load (once)
  useEffect(() => {
    if (!hasInitiallyScrolledRef.current && scrollContainerRef.current) {
      scrollToBottom();
      handleScroll();
      hasInitiallyScrolledRef.current = true;
    }
  }, [scrollToBottom, scrollContainerRef, handleScroll]);

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
    <EventStream
      events={events}
      isLoadingOlder={isLoadingOlder}
      hasReachedStart={isNewSession ? true : hasReachedStart}
      loadError={isNewSession ? null : paginationError}
      onRetryLoad={retryPagination}
      topSentinelRef={topSentinelRef}
      pendingToolCallIds={pendingToolCallIds}
      optimisticMessages={visibleOptimisticMessages}
      status={isNewSession ? undefined : status}
    />
  );
}

function AutopilotSessionEventsPageContent({
  loaderData,
}: Route.ComponentProps) {
  const { sessionId, eventsData, isNewSession } = loaderData;
  const { yoloMode, setYoloMode } = useAutopilotSession();
  const navigate = useNavigate();
  const { toast } = useToast();
  const interruptFetcher = useFetcher();

  // Track which session the interrupt was initiated for to prevent cross-session toast
  const interruptedSessionRef = useRef<string | null>(null);

  // Lift optimistic messages state to parent so ChatInput can work outside Suspense
  const [optimisticMessages, setOptimisticMessages] = useState<
    OptimisticMessage[]
  >([]);

  // Track autopilot status for disabling submit
  const [autopilotStatus, setAutopilotStatus] = useState<AutopilotStatus>({
    status: "idle",
  });

  const handleStatusChange = useCallback((status: AutopilotStatus) => {
    setAutopilotStatus(status);
  }, []);

  // Pending tool calls state - lifted from EventStreamContent for footer rendering
  const [pendingToolCalls, setPendingToolCalls] = useState<GatewayEvent[]>([]);

  // Derive pending tool call IDs Set once - used by EventStream and useAutoApproval
  const pendingToolCallIds = useMemo(
    () => new Set(pendingToolCalls.map((tc) => tc.id)),
    [pendingToolCalls],
  );

  const handlePendingToolCallsChange = useCallback(
    (toolCalls: GatewayEvent[]) => {
      setPendingToolCalls(toolCalls);
    },
    [],
  );

  // Derived values for queue-based approval UI
  const oldestPendingToolCall = pendingToolCalls[0] ?? null;

  // State for tool call authorization loading
  const [authLoadingStates, setAuthLoadingStates] = useState<
    Map<string, AuthorizationLoadingAction>
  >(new Map());

  // State for SSE connection error
  const [sseError, setSseError] = useState<{
    error: string | null;
    isRetrying: boolean;
  }>({ error: null, isRetrying: false });

  const handleErrorChange = useCallback(
    (error: string | null, isRetrying: boolean) => {
      setSseError({ error, isRetrying });
    },
    [],
  );

  // Track loading/error state for ChatInput - disabled until events resolve
  // For existing sessions, start loading until EventStreamContent calls onLoaded
  const [isEventsLoading, setIsEventsLoading] = useState(!isNewSession);
  const [hasLoadError, setHasLoadError] = useState(false);

  // Track whether we've reached the start of the conversation (for top fade)
  const [hasReachedStart, setHasReachedStart] = useState(false);

  const handleHasReachedStartChange = useCallback((reached: boolean) => {
    setHasReachedStart(reached);
  }, []);

  // Cooldown animation: triggers when the queue top changes due to SSE (not user action).
  // Covers both directions: new item jumping to top, or top item removed by external approval.
  // Does NOT trigger when queue was empty and first item arrives (no accidental click risk).
  const prevQueueTopRef = useRef<string | null>(null);
  const userActionRef = useRef(false);
  const [isInCooldown, setIsInCooldown] = useState(false);

  // Manual authorization hook - handles deduplication of authorization requests
  const manualAuthorization = useManualAuthorization(sessionId);

  // Reset loading/error state when navigating to a different session
  // Note: key={sessionId} on Suspense remounts EventStreamContent, which will call onLoaded
  useEffect(() => {
    setOptimisticMessages([]);
    setHasLoadError(false);
    setHasReachedStart(false);
    setAutopilotStatus({ status: "idle" });
    setPendingToolCalls([]);
    setAuthLoadingStates(new Map());
    setSseError({ error: null, isRetrying: false });
    prevQueueTopRef.current = null;
    manualAuthorization.reset();
    // Note: useAutoApproval handles its own cleanup on session change via internal effect
  }, [sessionId, isNewSession, manualAuthorization]);

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

  const { failedIds: failedAutoApprovals } = useAutoApproval({
    enabled: yoloMode && !isNewSession,
    sessionId,
    pendingToolCalls,
    pendingToolCallIds,
  });

  const handleApprove = useCallback(
    async (eventId: string) => {
      if (manualAuthorization.isProcessed(eventId)) return;
      userActionRef.current = true;
      setAuthLoadingStates((prev) => new Map(prev).set(eventId, "approving"));

      try {
        await manualAuthorization.approve(eventId);
      } catch {
        toast.error({
          title: "Approval failed",
          description: "Failed to submit approval. Please try again.",
        });
      } finally {
        setAuthLoadingStates((prev) => {
          const next = new Map(prev);
          next.delete(eventId);
          return next;
        });
      }
    },
    [manualAuthorization, toast],
  );

  const handleReject = useCallback(
    async (eventId: string) => {
      if (manualAuthorization.isProcessed(eventId)) return;
      userActionRef.current = true;
      setAuthLoadingStates((prev) => new Map(prev).set(eventId, "rejecting"));

      try {
        await manualAuthorization.reject(eventId);
      } catch {
        toast.error({
          title: "Rejection failed",
          description: "Failed to submit rejection. Please try again.",
        });
      } finally {
        setAuthLoadingStates((prev) => {
          const next = new Map(prev);
          next.delete(eventId);
          return next;
        });
      }
    },
    [manualAuthorization, toast],
  );

  const handleApproveAll = useCallback(async () => {
    if (pendingToolCalls.length === 0) return;

    const eventIds = pendingToolCalls.map((e) => e.id);
    const displayEventId = eventIds[0];
    const lastEventId = eventIds[eventIds.length - 1];

    // Early bailout if ALL events are already processed
    // (allows batch to proceed if some events are still pending)
    const hasUnprocessedEvents = eventIds.some(
      (id) => !manualAuthorization.isProcessed(id),
    );
    if (!hasUnprocessedEvents) return;

    userActionRef.current = true;
    setAuthLoadingStates((prev) =>
      new Map(prev).set(displayEventId, "approving_all"),
    );

    try {
      await manualAuthorization.approveAll(eventIds, lastEventId);
    } catch {
      toast.error({
        title: "Batch approval failed",
        description: "Failed to approve all tool calls. Please try again.",
      });
    } finally {
      setAuthLoadingStates((prev) => {
        const next = new Map(prev);
        next.delete(displayEventId);
        return next;
      });
    }
  }, [pendingToolCalls, manualAuthorization, toast]);

  // Handle interrupt session
  const handleInterruptSession = useCallback(() => {
    interruptedSessionRef.current = sessionId;
    interruptFetcher.submit(null, {
      method: "POST",
      action: `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/actions/interrupt`,
    });
  }, [interruptFetcher, sessionId]);

  // Show toast on interrupt result (only if still on the same session)
  useEffect(() => {
    if (interruptFetcher.state === "idle" && interruptFetcher.data) {
      // Only show toast if we're still on the session that was interrupted
      if (interruptedSessionRef.current !== sessionId) {
        return;
      }
      const data = interruptFetcher.data as {
        success: boolean;
        error?: string;
      };
      if (data.success) {
        toast.success({
          title: "Session interrupted",
          description: "The autopilot session has been interrupted.",
        });
      } else if (data.error) {
        toast.error({
          title: "Failed to interrupt session",
          description: data.error,
        });
      }
    }
  }, [interruptFetcher.state, interruptFetcher.data, toast, sessionId]);

  // Interruptible when actively processing (not idle or failed) and feature flag is enabled
  const { FF_INTERRUPT_SESSION } = useFeatureFlags();
  const isInterruptible =
    FF_INTERRUPT_SESSION &&
    autopilotStatus.status !== "idle" &&
    autopilotStatus.status !== "failed";

  // Disable submit unless status is idle or failed
  const submitDisabled =
    autopilotStatus.status !== "idle" && autopilotStatus.status !== "failed";

  const handleEventsLoaded = useCallback(() => {
    setIsEventsLoading(false);
    setHasLoadError(false);
  }, []);

  const handleLoadError = useCallback(() => {
    setIsEventsLoading(false);
    setHasLoadError(true);
  }, []);

  // Ref for scroll container - shared between parent and EventStreamContent
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  // Measure header/footer heights dynamically
  const [headerRef, headerHeight] = useElementHeight(56);
  const [footerRef, footerHeight] = useElementHeight(120);

  // State for fade overlays (both start false, updated on scroll)
  const [showTopFade, setShowTopFade] = useState(false);
  const [showBottomFade, setShowBottomFade] = useState(false);

  // Track previous footer height for scroll adjustment (null = initial mount)
  const prevFooterHeightRef = useRef<number | null>(null);

  // Reset footer height ref on session change to avoid cross-session scroll jumps
  useEffect(() => {
    prevFooterHeightRef.current = null;
  }, [sessionId]);

  // Adjust scroll position when footer height changes (e.g., tool card appears)
  // Only adjust if user is near bottom - don't disrupt users reading older messages
  useEffect(() => {
    const container = scrollContainerRef.current;

    // Skip initial mount - just record the value
    if (prevFooterHeightRef.current === null) {
      prevFooterHeightRef.current = footerHeight;
      return;
    }

    const delta = footerHeight - prevFooterHeightRef.current;
    prevFooterHeightRef.current = footerHeight;

    // Only adjust scroll when footer grows - shrinking doesn't need adjustment
    if (container && delta > 0) {
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      // When footer grows, distanceFromBottom increases by delta, so we need to
      // subtract it to get the user's original position before the change
      const originalDistance = distanceFromBottom - delta;
      const wasNearBottom = originalDistance < 100;

      if (wasNearBottom) {
        container.scrollTop += delta;
      }
    }
  }, [footerHeight]);

  // Update fade overlay visibility based on scroll position
  // Top fade stays visible until we've reached the start of the conversation
  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const target = e.currentTarget;
      setShowTopFade(target.scrollTop > 20 || !hasReachedStart);
      const distanceFromBottom =
        target.scrollHeight - target.scrollTop - target.clientHeight;
      setShowBottomFade(distanceFromBottom > 20);
    },
    [hasReachedStart],
  );

  // Update top fade when hasReachedStart changes (not just on scroll)
  // Without this, fade lingers if user is near top when reaching start
  useEffect(() => {
    if (hasReachedStart) {
      const container = scrollContainerRef.current;
      if (container && container.scrollTop <= 20) {
        setShowTopFade(false);
      }
    }
  }, [hasReachedStart]);

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
          tempId: uuid(),
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
    <div className="relative h-full">
      {/* Fixed header with breadcrumbs and fade gradient */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-20">
        <div className="container mx-auto px-8">
          {/* Header background - matches message width with slight outset */}
          <div ref={headerRef} className="bg-bg-secondary -mx-2 px-2 pt-4 pb-5">
            <div className="pointer-events-auto flex items-center justify-between">
              <Breadcrumbs
                segments={
                  isNewSession
                    ? [
                        { label: "Autopilot", href: "/autopilot/sessions" },
                        { label: "New Session" },
                      ]
                    : [
                        { label: "Autopilot", href: "/autopilot/sessions" },
                        { label: sessionId, isIdentifier: true },
                      ]
                }
              />
              <YoloModeToggle
                checked={yoloMode}
                onCheckedChange={setYoloMode}
              />
            </div>
            {sseError.error && sseError.isRetrying && (
              <AutopilotStatusBanner
                variant={AutopilotStatusBannerVariant.Warning}
                className="mt-3"
              >
                Failed to fetch events. Retrying...
              </AutopilotStatusBanner>
            )}
          </div>
          <FadeGradient
            direction={FadeDirection.Top}
            visible={showTopFade}
            className="-mx-2"
          />
        </div>
      </div>

      {/* Main scrollable area - full height with padding for header and footer */}
      <div
        ref={scrollContainerRef}
        className="h-full overflow-y-auto"
        onScroll={handleScroll}
      >
        <div
          className="container mx-auto px-8"
          style={{ paddingTop: headerHeight, paddingBottom: footerHeight }}
        >
          <Suspense fallback={<EventStreamSkeleton />}>
            <Await
              resolve={eventsData}
              errorElement={<EventStreamLoadError onError={handleLoadError} />}
            >
              {(resolvedData) => (
                <EventStreamContent
                  key={sessionId}
                  sessionId={sessionId}
                  eventsData={resolvedData}
                  isNewSession={isNewSession}
                  optimisticMessages={optimisticMessages}
                  onOptimisticMessagesChange={setOptimisticMessages}
                  scrollContainerRef={scrollContainerRef}
                  onLoaded={handleEventsLoaded}
                  onStatusChange={handleStatusChange}
                  onPendingToolCallsChange={handlePendingToolCallsChange}
                  onErrorChange={handleErrorChange}
                  onHasReachedStartChange={handleHasReachedStartChange}
                  pendingToolCallIds={pendingToolCallIds}
                />
              )}
            </Await>
          </Suspense>
        </div>
      </div>

      {/* Fixed footer with tool approval card and chat input */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20">
        <div className="container mx-auto px-8">
          <FadeGradient
            direction={FadeDirection.Bottom}
            visible={showBottomFade}
            className="-mx-2"
          />
          {/* Footer background - matches message width with slight outset */}
          <div ref={footerRef} className="bg-bg-secondary -mx-2 px-2">
            <div className="pointer-events-auto flex flex-col gap-4 pt-4 pb-8">
              {yoloMode && failedAutoApprovals.size > 0 && (
                <AutopilotStatusBanner
                  variant={AutopilotStatusBannerVariant.Warning}
                  icon={AlertTriangle}
                >
                  Auto-approval failed for {failedAutoApprovals.size} tool
                  {failedAutoApprovals.size === 1 ? " call" : " calls"}.
                  Retrying in background...
                </AutopilotStatusBanner>
              )}
              {oldestPendingToolCall && !yoloMode && (
                <PendingToolCallCard
                  key={oldestPendingToolCall.id}
                  event={oldestPendingToolCall}
                  isLoading={authLoadingStates.has(oldestPendingToolCall.id)}
                  loadingAction={authLoadingStates.get(
                    oldestPendingToolCall.id,
                  )}
                  onApprove={() => handleApprove(oldestPendingToolCall.id)}
                  onReject={() => handleReject(oldestPendingToolCall.id)}
                  onApproveAll={handleApproveAll}
                  additionalCount={pendingToolCalls.length - 1}
                  isInCooldown={isInCooldown}
                />
              )}
              <ChatInput
                sessionId={isNewSession ? NIL_UUID : sessionId}
                onMessageSent={handleMessageSent}
                onMessageFailed={handleMessageFailed}
                isNewSession={isNewSession}
                disabled={isEventsLoading || hasLoadError}
                submitDisabled={submitDisabled}
                isInterruptible={isInterruptible}
                isInterrupting={interruptFetcher.state !== "idle"}
                onInterrupt={handleInterruptSession}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AutopilotSessionEventsPage(
  props: Route.ComponentProps,
) {
  return (
    <AutopilotSessionProvider>
      <AutopilotSessionEventsPageContent {...props} />
    </AutopilotSessionProvider>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
