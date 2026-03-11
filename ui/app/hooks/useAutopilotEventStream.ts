import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AutopilotStatus,
  GatewayEvent,
  GatewayStreamUpdate,
} from "~/types/tensorzero";

interface UseAutopilotEventStreamOptions {
  sessionId: string;
  initialEvents: GatewayEvent[];
  initialPendingToolCalls: GatewayEvent[];
  initialPendingUserQuestions: GatewayEvent[];
  initialStatus: AutopilotStatus;
  enabled?: boolean;
}

interface UseAutopilotEventStreamResult {
  events: GatewayEvent[];
  pendingToolCalls: GatewayEvent[];
  pendingUserQuestions: GatewayEvent[];
  status: AutopilotStatus;
  isConnected: boolean;
  error: string | null;
  isRetrying: boolean;
  prependEvents: (newEvents: GatewayEvent[]) => void;
}

const RETRY_DELAY_MS = 5000;
const WHITELISTED_GRACE_PERIOD_MS = 5000;

/**
 * Compute which tool calls should be shown immediately (no grace period buffering).
 * - Tool calls that require approval are always shown immediately.
 * - Whitelisted tool calls (requires_approval === false) are only shown if they are
 *   older than the grace period, meaning auto-approval likely failed.
 */
function computeImmediateToolCalls(toolCalls: GatewayEvent[]): GatewayEvent[] {
  const now = Date.now();
  return toolCalls.filter((tc) => {
    if (tc.payload.type !== "tool_call") return false;
    if (tc.payload.requires_approval) return true;
    // Whitelisted: only show if older than grace period
    return (
      now - new Date(tc.created_at).getTime() >= WHITELISTED_GRACE_PERIOD_MS
    );
  });
}

type GraceBufferEntry = {
  event: GatewayEvent;
  timeoutId: ReturnType<typeof setTimeout>;
};

/**
 * Hook that streams autopilot events for a session.
 * Automatically reconnects on error with a 5-second delay.
 *
 * Whitelisted tool calls (requires_approval === false) are buffered for a grace
 * period before being shown. If an authorization/result event arrives during the
 * grace period, the tool call is silently discarded. Otherwise it becomes visible
 * for manual approval.
 */
export function useAutopilotEventStream({
  sessionId,
  initialEvents,
  initialPendingToolCalls,
  initialPendingUserQuestions,
  initialStatus,
  enabled = true,
}: UseAutopilotEventStreamOptions): UseAutopilotEventStreamResult {
  const [events, setEvents] = useState<GatewayEvent[]>(initialEvents);
  const [pendingToolCalls, setPendingToolCalls] = useState<GatewayEvent[]>(() =>
    computeImmediateToolCalls(initialPendingToolCalls),
  );
  const [pendingUserQuestions, setPendingUserQuestions] = useState<
    GatewayEvent[]
  >(initialPendingUserQuestions);
  const [status, setStatus] = useState<AutopilotStatus>(initialStatus);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);

  // Grace period buffer for whitelisted tool calls
  const graceBufferRef = useRef<Map<string, GraceBufferEntry>>(new Map());

  // Track the last event ID for reconnection
  const lastEventIdRef = useRef<string | null>(
    initialEvents.length > 0
      ? initialEvents[initialEvents.length - 1].id
      : null,
  );

  // Track if component is mounted
  const isMountedRef = useRef(true);

  // AbortController for cleanup
  const abortControllerRef = useRef<AbortController | null>(null);

  // Retry timeout ref
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Grace period helpers
  const clearAllGraceTimers = useCallback(() => {
    for (const entry of graceBufferRef.current.values()) {
      clearTimeout(entry.timeoutId);
    }
    graceBufferRef.current.clear();
  }, []);

  const cancelGraceTimer = useCallback((eventId: string) => {
    const entry = graceBufferRef.current.get(eventId);
    if (entry) {
      clearTimeout(entry.timeoutId);
      graceBufferRef.current.delete(eventId);
    }
  }, []);

  const promoteFromGraceBuffer = useCallback((eventId: string) => {
    const entry = graceBufferRef.current.get(eventId);
    if (!entry) return;
    graceBufferRef.current.delete(eventId);

    setPendingToolCalls((prev) => {
      if (prev.some((e) => e.id === eventId)) return prev;
      return [...prev, entry.event].sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
    });
  }, []);

  const bufferWhitelistedToolCall = useCallback(
    (event: GatewayEvent, delayMs: number) => {
      // Don't buffer if already in buffer or already in pending
      if (graceBufferRef.current.has(event.id)) return;

      const timeoutId = setTimeout(() => {
        promoteFromGraceBuffer(event.id);
      }, delayMs);

      graceBufferRef.current.set(event.id, { event, timeoutId });
    },
    [promoteFromGraceBuffer],
  );

  const connect = useCallback(async () => {
    if (!enabled || !isMountedRef.current) return;

    // Cancel any pending retry
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }

    // Abort any existing connection
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const url = new URL(
      `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events/stream`,
      window.location.origin,
    );

    if (lastEventIdRef.current) {
      url.searchParams.set("last_event_id", lastEventIdRef.current);
    }

    try {
      setError(null);
      setIsRetrying(false);

      const response = await fetch(url.toString(), {
        headers: { Accept: "text/event-stream" },
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      if (!isMountedRef.current) return;
      setIsConnected(true);

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Response body is not readable");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done || !isMountedRef.current) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6).trim();
              if (data) {
                try {
                  const streamUpdate = JSON.parse(data) as GatewayStreamUpdate;
                  const event = streamUpdate.event;
                  lastEventIdRef.current = event.id;

                  setEvents((prev) => {
                    // Check if event already exists (avoid duplicates)
                    if (prev.some((e) => e.id === event.id)) {
                      return prev;
                    }
                    // Insert in sorted order by created_at
                    const newEvents = [...prev, event].sort(
                      (a, b) =>
                        new Date(a.created_at).getTime() -
                        new Date(b.created_at).getTime(),
                    );
                    return newEvents;
                  });

                  // Handle authorization/result events: cancel grace timers and remove from pending
                  if (
                    event.payload.type === "tool_call_authorization" ||
                    event.payload.type === "tool_result"
                  ) {
                    const toolCallEventId = event.payload.tool_call_event_id;
                    cancelGraceTimer(toolCallEventId);
                    setPendingToolCalls((prev) =>
                      prev.filter((e) => e.id !== toolCallEventId),
                    );
                  }

                  // Handle new tool call events
                  if (event.payload.type === "tool_call") {
                    if (event.payload.requires_approval) {
                      // Requires approval: add to pending immediately
                      setPendingToolCalls((prev) => {
                        if (prev.some((e) => e.id === event.id)) return prev;
                        return [...prev, event].sort(
                          (a, b) =>
                            new Date(a.created_at).getTime() -
                            new Date(b.created_at).getTime(),
                        );
                      });
                    } else {
                      // Whitelisted: buffer with grace period
                      bufferWhitelistedToolCall(
                        event,
                        WHITELISTED_GRACE_PERIOD_MS,
                      );
                    }
                  }

                  // Update pending user questions based on event type
                  // Both user_questions and auto_eval_example_labeling events
                  // require user input and share the same pending queue.
                  setPendingUserQuestions((prev) => {
                    if (
                      event.payload.type === "user_questions" ||
                      event.payload.type === "auto_eval_example_labeling"
                    ) {
                      if (prev.some((e) => e.id === event.id)) {
                        return prev;
                      }
                      return [...prev, event].sort(
                        (a, b) =>
                          new Date(a.created_at).getTime() -
                          new Date(b.created_at).getTime(),
                      );
                    }
                    if (event.payload.type === "user_questions_answers") {
                      const questionEventId =
                        event.payload.user_questions_event_id;
                      return prev.filter((e) => e.id !== questionEventId);
                    }
                    return prev;
                  });

                  // Update autopilot status
                  setStatus(streamUpdate.status);
                } catch {
                  // Skip invalid JSON
                }
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      // Stream ended normally, reconnect to get more events
      if (isMountedRef.current && !abortController.signal.aborted) {
        setIsConnected(false);
        // Small delay before reconnecting after normal stream end
        retryTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            connect();
          }
        }, 1000);
      }
    } catch (err) {
      if (!isMountedRef.current) return;

      // Don't show error for intentional aborts
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }

      const errorMessage =
        err instanceof Error ? err.message : "Connection failed";

      setIsConnected(false);
      setError(errorMessage);
      setIsRetrying(true);

      // Schedule retry
      retryTimeoutRef.current = setTimeout(() => {
        if (isMountedRef.current) {
          connect();
        }
      }, RETRY_DELAY_MS);
    }
  }, [sessionId, enabled, cancelGraceTimer, bufferWhitelistedToolCall]);

  useEffect(() => {
    isMountedRef.current = true;

    if (enabled) {
      connect();
    }

    return () => {
      isMountedRef.current = false;

      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      clearAllGraceTimers();
    };
  }, [connect, enabled, clearAllGraceTimers]);

  // Update events when initialEvents change (e.g., on page refresh)
  useEffect(() => {
    clearAllGraceTimers();

    setEvents(initialEvents);
    setPendingUserQuestions(initialPendingUserQuestions);
    setStatus(initialStatus);
    lastEventIdRef.current =
      initialEvents.length > 0
        ? initialEvents[initialEvents.length - 1].id
        : null;

    // Process initial pending tool calls with grace period logic
    const now = Date.now();
    const immediate: GatewayEvent[] = [];

    for (const tc of initialPendingToolCalls) {
      if (tc.payload.type !== "tool_call") continue;
      if (tc.payload.requires_approval) {
        immediate.push(tc);
      } else {
        const age = now - new Date(tc.created_at).getTime();
        if (age >= WHITELISTED_GRACE_PERIOD_MS) {
          immediate.push(tc);
        } else {
          bufferWhitelistedToolCall(tc, WHITELISTED_GRACE_PERIOD_MS - age);
        }
      }
    }

    setPendingToolCalls(
      immediate.sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      ),
    );
  }, [
    initialEvents,
    initialPendingToolCalls,
    initialPendingUserQuestions,
    initialStatus,
    clearAllGraceTimers,
    bufferWhitelistedToolCall,
  ]);

  // Allow prepending older events (for reverse infinite scroll)
  const prependEvents = useCallback((newEvents: GatewayEvent[]) => {
    setEvents((prev) => {
      // Create a set of existing event IDs for deduplication
      const existingIds = new Set(prev.map((e) => e.id));

      // Filter out duplicates and merge
      const uniqueNewEvents = newEvents.filter((e) => !existingIds.has(e.id));

      // Combine and sort by created_at
      return [...uniqueNewEvents, ...prev].sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
    });
  }, []);

  return {
    events,
    pendingToolCalls,
    pendingUserQuestions,
    status,
    isConnected,
    error,
    isRetrying,
    prependEvents,
  };
}
