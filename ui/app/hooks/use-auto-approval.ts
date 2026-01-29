import { useCallback, useEffect, useRef, useState } from "react";

interface UseAutoApprovalOptions {
  enabled: boolean;
  sessionId: string;
  pendingToolCallIds: string[];
}

interface UseAutoApprovalResult {
  failedIds: Set<string>;
  reset: () => void;
}

async function authorizeToolCall(
  sessionId: string,
  eventId: string,
): Promise<void> {
  const response = await fetch(
    `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events/authorize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_call_event_id: eventId,
        status: { type: "approved" },
      }),
    },
  );
  if (!response.ok) {
    throw new Error("Authorization failed");
  }
}

/**
 * Hook to handle auto-approval of tool calls with exponential backoff retry.
 *
 * Retry schedule: 1s, 2s, 4s, then 60s forever after showing error.
 * After 3 retries, the tool call ID is added to `failedIds`.
 */
export function useAutoApproval({
  enabled,
  sessionId,
  pendingToolCallIds,
}: UseAutoApprovalOptions): UseAutoApprovalResult {
  // Refs to access current values in async callbacks (avoid stale closures)
  const enabledRef = useRef(enabled);
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  const pendingToolCallIdsRef = useRef(pendingToolCallIds);
  useEffect(() => {
    pendingToolCallIdsRef.current = pendingToolCallIds;
  }, [pendingToolCallIds]);

  const retryCountsRef = useRef<Map<string, number>>(new Map());
  const retryTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const inFlightRef = useRef<Set<string>>(new Set());

  const [failedIds, setFailedIds] = useState<Set<string>>(new Set());

  const reset = useCallback(() => {
    for (const timer of retryTimersRef.current.values()) {
      clearTimeout(timer);
    }
    retryTimersRef.current.clear();
    retryCountsRef.current.clear();
    inFlightRef.current.clear();
    setFailedIds(new Set());
  }, []);

  useEffect(() => {
    const pendingSet = new Set(pendingToolCallIds);

    for (const [eventId, timer] of retryTimersRef.current) {
      if (!pendingSet.has(eventId)) {
        clearTimeout(timer);
        retryTimersRef.current.delete(eventId);
        retryCountsRef.current.delete(eventId);
      }
    }

    for (const eventId of inFlightRef.current) {
      if (!pendingSet.has(eventId)) {
        inFlightRef.current.delete(eventId);
        retryCountsRef.current.delete(eventId);
      }
    }

    setFailedIds((prev) => {
      const next = new Set(prev);
      let changed = false;
      for (const eventId of prev) {
        if (!pendingSet.has(eventId)) {
          next.delete(eventId);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [pendingToolCallIds]);

  useEffect(() => {
    if (!enabled) {
      reset();
    }
  }, [enabled, reset]);

  // Full cleanup on unmount
  useEffect(() => {
    return reset;
  }, [reset]);

  // Exponential backoff: 1s, 2s, 4s, then 60s forever
  const getRetryDelay = (retryCount: number): number => {
    if (retryCount < 3) {
      return 1000 * Math.pow(2, retryCount);
    }
    return 60000;
  };

  const attemptApproval = useCallback((eventId: string) => {
    inFlightRef.current.add(eventId);

    authorizeToolCall(sessionIdRef.current, eventId).then(
      () => {
        inFlightRef.current.delete(eventId);
      },
      () => {
        inFlightRef.current.delete(eventId);

        if (!enabledRef.current) return;
        if (!pendingToolCallIdsRef.current.includes(eventId)) return;

        const retryCount = (retryCountsRef.current.get(eventId) ?? 0) + 1;
        retryCountsRef.current.set(eventId, retryCount);

        if (retryCount === 3) {
          setFailedIds((prev) => new Set(prev).add(eventId));
        }

        const delay = getRetryDelay(retryCount);
        const timer = setTimeout(() => {
          retryTimersRef.current.delete(eventId);

          if (!enabledRef.current) return;
          if (!pendingToolCallIdsRef.current.includes(eventId)) return;

          attemptApproval(eventId);
        }, delay);
        retryTimersRef.current.set(eventId, timer);
      },
    );
  }, []);

  useEffect(() => {
    if (!enabled || pendingToolCallIds.length === 0) return;

    for (const eventId of pendingToolCallIds) {
      if (
        inFlightRef.current.has(eventId) ||
        retryTimersRef.current.has(eventId)
      ) {
        continue;
      }
      attemptApproval(eventId);
    }
  }, [enabled, pendingToolCallIds, attemptApproval]);

  return { failedIds, reset };
}
