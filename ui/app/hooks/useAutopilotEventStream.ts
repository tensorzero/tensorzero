import { useCallback, useEffect, useRef, useState } from "react";
import type { Event } from "~/types/tensorzero";

interface UseAutopilotEventStreamOptions {
  sessionId: string;
  initialEvents: Event[];
  initialPendingToolCalls: Event[];
  enabled?: boolean;
}

interface UseAutopilotEventStreamResult {
  events: Event[];
  pendingToolCalls: Event[];
  isConnected: boolean;
  error: string | null;
  isRetrying: boolean;
  prependEvents: (newEvents: Event[]) => void;
}

const RETRY_DELAY_MS = 5000;

/**
 * Hook that streams autopilot events for a session.
 * Automatically reconnects on error with a 5-second delay.
 */
export function useAutopilotEventStream({
  sessionId,
  initialEvents,
  initialPendingToolCalls,
  enabled = true,
}: UseAutopilotEventStreamOptions): UseAutopilotEventStreamResult {
  const [events, setEvents] = useState<Event[]>(initialEvents);
  const [pendingToolCalls, setPendingToolCalls] = useState<Event[]>(
    initialPendingToolCalls,
  );
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);

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
                  const event = JSON.parse(data) as Event;
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

                  // Update pending tool calls based on event type
                  setPendingToolCalls((prev) => {
                    if (event.payload.type === "tool_call") {
                      // Add new tool call if not already present
                      if (prev.some((e) => e.id === event.id)) {
                        return prev;
                      }
                      return [...prev, event].sort(
                        (a, b) =>
                          new Date(a.created_at).getTime() -
                          new Date(b.created_at).getTime(),
                      );
                    }
                    if (
                      event.payload.type === "tool_call_authorization" ||
                      event.payload.type === "tool_result"
                    ) {
                      // Remove tool call that was authorized or got a result
                      const toolCallEventId = event.payload.tool_call_event_id;
                      return prev.filter((e) => e.id !== toolCallEventId);
                    }
                    return prev;
                  });
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
  }, [sessionId, enabled]);

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
    };
  }, [connect, enabled]);

  // Update events when initialEvents change (e.g., on page refresh)
  useEffect(() => {
    setEvents(initialEvents);
    setPendingToolCalls(initialPendingToolCalls);
    lastEventIdRef.current =
      initialEvents.length > 0
        ? initialEvents[initialEvents.length - 1].id
        : null;
  }, [initialEvents, initialPendingToolCalls]);

  // Allow prepending older events (for reverse infinite scroll)
  const prependEvents = useCallback((newEvents: Event[]) => {
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
    isConnected,
    error,
    isRetrying,
    prependEvents,
  };
}
