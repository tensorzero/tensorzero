import { useCallback, useEffect, useRef, useState } from "react";

interface UseAutoApprovalOptions {
  enabled: boolean;
  sessionId: string;
  pendingToolCallIds: string[];
}

interface UseAutoApprovalResult {
  failedIds: Set<string>;
}

async function approveAllToolCalls(
  sessionId: string,
  lastToolCallEventId: string,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/actions/approve_all`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        last_tool_call_event_id: lastToolCallEventId,
      }),
      signal,
    },
  );
  if (!response.ok) {
    throw new Error("Batch approval failed");
  }
}

/**
 * Hook to handle auto-approval of tool calls using batch approval with retry.
 *
 * Uses approve_all endpoint to batch-approve all pending tool calls up to
 * the highest pending ID.
 *
 * Retry schedule: 1s, 2s, 4s, then 60s forever after showing error.
 * After 3 retries, all pending tool call IDs are added to `failedIds`.
 *
 * Requests are cancelled when disabled, session changes, or on unmount.
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

  const pendingToolCallIdsRef = useRef(pendingToolCallIds);
  useEffect(() => {
    pendingToolCallIdsRef.current = pendingToolCallIds;
  }, [pendingToolCallIds]);

  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<NodeJS.Timeout | null>(null);
  const inFlightRef = useRef(false);
  const lastAttemptedIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [failedIds, setFailedIds] = useState<Set<string>>(new Set());

  // Clear timers and tracking state (does not touch AbortController)
  const clearState = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    retryCountRef.current = 0;
    inFlightRef.current = false;
    lastAttemptedIdRef.current = null;
    setFailedIds(new Set());
  }, []);

  // Cleanup when pending tool calls change (tool calls resolved)
  useEffect(() => {
    const pendingSet = new Set(pendingToolCallIds);

    // If the ID we were trying to approve is no longer pending, it succeeded
    // Clear the retry state
    if (
      lastAttemptedIdRef.current &&
      !pendingSet.has(lastAttemptedIdRef.current)
    ) {
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      retryCountRef.current = 0;
      lastAttemptedIdRef.current = null;
    }

    // Update failedIds - remove any that are no longer pending
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

  // Manage AbortController lifecycle
  useEffect(() => {
    if (!enabled) {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      clearState();
      return;
    }

    abortControllerRef.current = new AbortController();

    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
    };
  }, [enabled, sessionId, clearState]);

  // Exponential backoff: 1s, 2s, 4s, then 60s forever
  const getRetryDelay = (retryCount: number): number => {
    if (retryCount < 3) {
      return 1000 * Math.pow(2, retryCount);
    }
    return 60000;
  };

  const attemptBatchApproval = useCallback(() => {
    const controller = abortControllerRef.current;
    if (!controller) return;

    const currentPendingIds = pendingToolCallIdsRef.current;
    if (currentPendingIds.length === 0) return;

    // Get the most recent pending ID - API will approve all pending calls up to this ID
    const lastId = currentPendingIds[currentPendingIds.length - 1];

    inFlightRef.current = true;
    lastAttemptedIdRef.current = lastId;

    approveAllToolCalls(sessionId, lastId, controller.signal).then(
      () => {
        inFlightRef.current = false;
        retryCountRef.current = 0;
      },
      (error: Error) => {
        inFlightRef.current = false;

        if (error.name === "AbortError") return;
        if (!enabledRef.current) return;

        const newRetryCount = retryCountRef.current + 1;
        retryCountRef.current = newRetryCount;

        // After 3 failures, mark all current pending IDs as failed
        if (newRetryCount === 3) {
          setFailedIds(new Set(pendingToolCallIdsRef.current));
        }

        const delay = getRetryDelay(newRetryCount);
        retryTimerRef.current = setTimeout(() => {
          retryTimerRef.current = null;

          if (!enabledRef.current) return;
          if (pendingToolCallIdsRef.current.length === 0) return;

          attemptBatchApproval();
        }, delay);
      },
    );
  }, [sessionId]);

  // Trigger batch approval when pending IDs change
  useEffect(() => {
    if (!enabled || pendingToolCallIds.length === 0) return;

    // Don't start a new attempt if one is in-flight or scheduled
    if (inFlightRef.current || retryTimerRef.current) return;

    attemptBatchApproval();
  }, [enabled, pendingToolCallIds, attemptBatchApproval]);

  return { failedIds };
}
