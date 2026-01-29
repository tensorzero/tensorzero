import { useCallback, useEffect, useRef, useState } from "react";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { useLatest } from "./use-latest";

// =============================================================================
// Configuration
// =============================================================================

const RETRY_BASE_DELAY_MS = 1000;
const RETRY_MAX_DELAY_MS = 60000;
const RETRY_SHOW_ERROR_THRESHOLD = 3;

// =============================================================================
// Types
// =============================================================================

interface UseAutoApprovalOptions {
  enabled: boolean;
  sessionId: string;
  pendingToolCalls: { id: string }[];
  pendingToolCallIds: Set<string>;
}

interface UseAutoApprovalResult {
  failedIds: Set<string>;
}

// =============================================================================
// Pure Helpers (no hooks/refs)
// =============================================================================

/**
 * Calculate retry delay with exponential backoff.
 * Schedule: 1s, 2s, 4s, then 60s forever after threshold.
 */
function calculateRetryDelay(retryCount: number): number {
  if (retryCount < RETRY_SHOW_ERROR_THRESHOLD) {
    return RETRY_BASE_DELAY_MS * Math.pow(2, retryCount);
  }
  return RETRY_MAX_DELAY_MS;
}

/**
 * Remove resolved IDs from the failed set.
 * Returns new Set if changed, same reference if unchanged (for React state).
 */
function pruneResolvedFromFailedIds(
  failedIds: Set<string>,
  pendingToolCallIds: Set<string>,
): Set<string> {
  const pruned = new Set(failedIds);
  let changed = false;
  for (const eventId of failedIds) {
    if (!pendingToolCallIds.has(eventId)) {
      pruned.delete(eventId);
      changed = true;
    }
  }
  return changed ? pruned : failedIds;
}

// =============================================================================
// Hook
// =============================================================================

/**
 * Auto-approve tool calls using batch approval with retry.
 *
 * Flow:
 * ┌───────┐  pending arrives   ┌───────────┐  success   ┌───────┐
 * │ Idle  │ ─────────────────► │ Attempting│ ──────────►│ Idle  │
 * └───────┘                    └─────┬─────┘            └───────┘
 *                                    │ failure
 *                                    ▼
 *                             ┌─────────────┐  retry
 *                             │ Retrying    │ ──────┐
 *                             └──────┬──────┘       │
 *                                    │ 3 failures   │
 *                                    ▼              │
 *                             ┌──────────┐          │
 *                             │ Failed   │ ◄────────┘
 *                             │(show err)│
 *                             └──────────┘
 *
 * Retry schedule: 1s → 2s → 4s → 60s (forever)
 * After 3 failures, all pending IDs are marked as failed (shown in UI).
 * Requests are cancelled when disabled, session changes, or on unmount.
 */
export function useAutoApproval({
  enabled,
  sessionId,
  pendingToolCalls,
  pendingToolCallIds,
}: UseAutoApprovalOptions): UseAutoApprovalResult {
  // ---------------------------------------------------------------------------
  // Refs for async callback access (avoid stale closures)
  // ---------------------------------------------------------------------------
  const enabledRef = useLatest(enabled);
  const pendingToolCallsRef = useLatest(pendingToolCalls);
  const pendingToolCallIdsRef = useLatest(pendingToolCallIds);

  // ---------------------------------------------------------------------------
  // Internal state
  // ---------------------------------------------------------------------------
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<NodeJS.Timeout | null>(null);
  const inFlightRef = useRef(false);
  const lastAttemptedIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [failedIds, setFailedIds] = useState<Set<string>>(new Set());

  // ---------------------------------------------------------------------------
  // State management helpers
  // ---------------------------------------------------------------------------

  /** Clear retry timer if one is scheduled. */
  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  /** Reset all tracking state (timers, counters, flags). Does not touch AbortController. */
  const resetTrackingState = useCallback(() => {
    clearRetryTimer();
    retryCountRef.current = 0;
    inFlightRef.current = false;
    lastAttemptedIdRef.current = null;
    setFailedIds(new Set());
  }, [clearRetryTimer]);

  /** Clear retry state when the tool call we were attempting has been resolved. */
  const clearRetryStateIfAttemptResolved = useCallback(() => {
    if (
      lastAttemptedIdRef.current &&
      !pendingToolCallIds.has(lastAttemptedIdRef.current)
    ) {
      clearRetryTimer();
      retryCountRef.current = 0;
      lastAttemptedIdRef.current = null;
    }
  }, [pendingToolCallIds, clearRetryTimer]);

  // ---------------------------------------------------------------------------
  // Approval logic
  // ---------------------------------------------------------------------------

  const attemptBatchApproval = useCallback(() => {
    const controller = abortControllerRef.current;
    if (!controller) return;

    const currentPending = pendingToolCallsRef.current;
    if (currentPending.length === 0) return;

    // Approve all pending calls up to the most recent ID
    const lastId = currentPending[currentPending.length - 1].id;

    inFlightRef.current = true;
    lastAttemptedIdRef.current = lastId;

    const handleSuccess = () => {
      inFlightRef.current = false;
      retryCountRef.current = 0;
    };

    const handleFailure = (error: Error) => {
      inFlightRef.current = false;

      // Ignore aborted requests or if disabled while in-flight
      if (error.name === "AbortError") return;
      if (!enabledRef.current) return;

      const newRetryCount = retryCountRef.current + 1;
      retryCountRef.current = newRetryCount;

      // After threshold failures, mark all current pending IDs as failed
      if (newRetryCount === RETRY_SHOW_ERROR_THRESHOLD) {
        setFailedIds(new Set(pendingToolCallIdsRef.current));
      }

      // Schedule retry with exponential backoff
      const delay = calculateRetryDelay(newRetryCount);
      retryTimerRef.current = setTimeout(() => {
        retryTimerRef.current = null;

        // Guard: could have been disabled or resolved during delay
        if (!enabledRef.current) return;
        if (pendingToolCallsRef.current.length === 0) return;

        attemptBatchApproval();
      }, delay);
    };

    approveAllToolCalls(sessionId, lastId, controller.signal).then(
      handleSuccess,
      handleFailure,
    );
  }, [sessionId, enabledRef, pendingToolCallsRef, pendingToolCallIdsRef]);

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Effect: Clean up when tool calls are resolved (success detected via SSE)
  useEffect(() => {
    clearRetryStateIfAttemptResolved();
    setFailedIds((prev) =>
      pruneResolvedFromFailedIds(prev, pendingToolCallIds),
    );
  }, [pendingToolCallIds, clearRetryStateIfAttemptResolved]);

  // Effect: Manage AbortController lifecycle (enabled/session changes)
  useEffect(() => {
    if (!enabled) {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      resetTrackingState();
      return;
    }

    abortControllerRef.current = new AbortController();

    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      resetTrackingState();
    };
  }, [enabled, sessionId, resetTrackingState]);

  // Effect: Trigger approval when new pending tool calls arrive
  useEffect(() => {
    if (!enabled || pendingToolCalls.length === 0) return;

    // Don't start a new attempt if one is in-flight or scheduled
    if (inFlightRef.current || retryTimerRef.current) return;

    attemptBatchApproval();
  }, [enabled, pendingToolCalls, attemptBatchApproval]);

  return { failedIds };
}
