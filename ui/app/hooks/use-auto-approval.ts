import { useCallback, useEffect, useRef, useState } from "react";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { useLatest } from "./use-latest";

// Retry configuration
const RETRY_BASE_DELAY_MS = 1000;
const RETRY_MAX_DELAY_MS = 60000;
const RETRY_SHOW_ERROR_THRESHOLD = 3;

interface UseAutoApprovalOptions {
  enabled: boolean;
  sessionId: string;
  pendingToolCalls: { id: string }[];
  pendingToolCallIds: Set<string>;
}

interface UseAutoApprovalResult {
  failedIds: Set<string>;
}

/**
 * Hook to handle auto-approval of tool calls using batch approval with retry.
 *
 * Uses approve_all endpoint to batch-approve all pending tool calls up to
 * the most recent pending ID.
 *
 * Retry schedule: 1s, 2s, 4s, then 60s forever after showing error.
 * After RETRY_SHOW_ERROR_THRESHOLD failures, all pending IDs are added to `failedIds`.
 *
 * Requests are cancelled when disabled, session changes, or on unmount.
 */
export function useAutoApproval({
  enabled,
  sessionId,
  pendingToolCalls,
  pendingToolCallIds,
}: UseAutoApprovalOptions): UseAutoApprovalResult {
  // Refs to access current values in async callbacks (avoid stale closures)
  const enabledRef = useLatest(enabled);
  const pendingToolCallsRef = useLatest(pendingToolCalls);
  const pendingToolCallIdsRef = useLatest(pendingToolCallIds);

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
    // If the ID we were trying to approve is no longer pending, it succeeded
    // Clear the retry state
    if (
      lastAttemptedIdRef.current &&
      !pendingToolCallIds.has(lastAttemptedIdRef.current)
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
        if (!pendingToolCallIds.has(eventId)) {
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
      clearState();
    };
  }, [enabled, sessionId, clearState]);

  // Exponential backoff: 1s, 2s, 4s, then 60s forever
  const getRetryDelay = (retryCount: number): number => {
    if (retryCount < RETRY_SHOW_ERROR_THRESHOLD) {
      return RETRY_BASE_DELAY_MS * Math.pow(2, retryCount);
    }
    return RETRY_MAX_DELAY_MS;
  };

  const attemptBatchApproval = useCallback(() => {
    const controller = abortControllerRef.current;
    if (!controller) return;

    const currentPending = pendingToolCallsRef.current;
    if (currentPending.length === 0) return;

    // Get the most recent pending ID - API will approve all pending calls up to this ID
    const lastId = currentPending[currentPending.length - 1].id;

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

        // After threshold failures, mark all current pending IDs as failed
        if (newRetryCount === RETRY_SHOW_ERROR_THRESHOLD) {
          setFailedIds(new Set(pendingToolCallIdsRef.current));
        }

        const delay = getRetryDelay(newRetryCount);
        retryTimerRef.current = setTimeout(() => {
          retryTimerRef.current = null;

          if (!enabledRef.current) return;
          if (pendingToolCallsRef.current.length === 0) return;

          attemptBatchApproval();
        }, delay);
      },
    );
  }, [sessionId, enabledRef, pendingToolCallsRef, pendingToolCallIdsRef]);

  // Trigger batch approval when pending IDs change
  useEffect(() => {
    if (!enabled || pendingToolCalls.length === 0) return;

    // Don't start a new attempt if one is in-flight or scheduled
    if (inFlightRef.current || retryTimerRef.current) return;

    attemptBatchApproval();
  }, [enabled, pendingToolCalls, attemptBatchApproval]);

  return { failedIds };
}
