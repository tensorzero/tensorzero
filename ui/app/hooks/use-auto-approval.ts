import { useCallback, useEffect, useReducer, useRef } from "react";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { useLatest } from "./use-latest";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const RETRY_BASE_DELAY_MS = 1000;
const RETRY_MAX_DELAY_MS = 60000;
const RETRY_SHOW_ERROR_THRESHOLD = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface UseAutoApprovalOptions {
  enabled: boolean;
  sessionId: string;
  pendingToolCalls: { id: string }[];
  pendingToolCallIds: Set<string>;
}

interface UseAutoApprovalResult {
  failedIds: Set<string>;
}

// ─────────────────────────────────────────────────────────────────────────────
// State Machine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Approval states:
 * - idle: Ready to approve, no pending request
 * - attempting: Request in flight
 * - waiting_retry: Failed, auto-retry scheduled
 * - failed: Exceeded retry threshold, showing error in UI
 */
type ApprovalStatus = "idle" | "attempting" | "waiting_retry" | "failed";

type ApprovalState = {
  status: ApprovalStatus;
  retryCount: number;
  attemptingId: string | null;
  failedIds: Set<string>;
};

type ApprovalAction =
  | { type: "START_ATTEMPT"; id: string }
  | { type: "ATTEMPT_SUCCESS" }
  | { type: "SCHEDULE_RETRY" }
  | { type: "MAX_RETRIES_EXCEEDED"; failedIds: Set<string> }
  | { type: "TOOL_CALLS_RESOLVED"; pendingIds: Set<string> }
  | { type: "RESET" };

function approvalReducer(
  state: ApprovalState,
  action: ApprovalAction,
): ApprovalState {
  switch (action.type) {
    case "START_ATTEMPT":
      return {
        ...state,
        status: "attempting",
        attemptingId: action.id,
      };

    case "ATTEMPT_SUCCESS":
      return {
        ...state,
        status: "idle",
        retryCount: 0,
        attemptingId: null,
      };

    case "SCHEDULE_RETRY":
      return {
        ...state,
        status: "waiting_retry",
        retryCount: state.retryCount + 1,
      };

    case "MAX_RETRIES_EXCEEDED":
      return {
        ...state,
        status: "failed",
        retryCount: state.retryCount + 1,
        failedIds: action.failedIds,
      };

    case "TOOL_CALLS_RESOLVED": {
      // Clear attemptingId if it was resolved
      const attemptingResolved =
        state.attemptingId && !action.pendingIds.has(state.attemptingId);

      // Prune failedIds - remove any that are no longer pending
      const prunedFailedIds = new Set(state.failedIds);
      let failedChanged = false;
      for (const id of state.failedIds) {
        if (!action.pendingIds.has(id)) {
          prunedFailedIds.delete(id);
          failedChanged = true;
        }
      }

      if (!attemptingResolved && !failedChanged) {
        return state; // No change
      }

      return {
        ...state,
        status: attemptingResolved ? "idle" : state.status,
        retryCount: attemptingResolved ? 0 : state.retryCount,
        attemptingId: attemptingResolved ? null : state.attemptingId,
        failedIds: failedChanged ? prunedFailedIds : state.failedIds,
      };
    }

    case "RESET":
      return createInitialState();

    default:
      return state;
  }
}

function createInitialState(): ApprovalState {
  return {
    status: "idle",
    retryCount: 0,
    attemptingId: null,
    failedIds: new Set(),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/** Check if we can start a new approval attempt. */
function canStartAttempt(status: ApprovalStatus): boolean {
  // Can start from idle, waiting_retry, or failed (keeps retrying in background)
  // Only "attempting" blocks (request already in flight)
  return status !== "attempting";
}

/** Calculate retry delay with exponential backoff: 1s, 2s, 4s, then 60s forever. */
function getRetryDelay(retryCount: number): number {
  if (retryCount < RETRY_SHOW_ERROR_THRESHOLD) {
    return RETRY_BASE_DELAY_MS * Math.pow(2, retryCount);
  }
  return RETRY_MAX_DELAY_MS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Auto-approve tool calls using batch approval with retry.
 *
 * ## How It Works
 *
 * 1. **Trigger**: New pending tool calls arrive via SSE
 * 2. **Guard**: Check state synchronously via ref to prevent concurrent attempts
 * 3. **Approve**: Call approve_all API for all pending up to newest ID
 * 4. **Retry**: On failure, exponential backoff (1s → 2s → 4s → 60s)
 * 5. **Error**: After 3 failures, show error banner (keeps retrying in background)
 *
 * ## State Machine
 *
 * ```
 *      ┌─────────────────────────────────────────────────────────────┐
 *      │                                                             │
 *      ▼                                                             │
 *    idle ──[START_ATTEMPT]──► attempting                            │
 *      ▲                           │                                 │
 *      │            ┌──────────────┴──────────────┐                  │
 *      │            ▼                             ▼                  │
 *      │   [ATTEMPT_SUCCESS]                 [on error]              │
 *      │           │                              │                  │
 *      │           ▼                    ┌─────────┴─────────┐        │
 *      │         idle                   ▼                   ▼        │
 *      │                          waiting_retry          failed      │
 *      │                         (retries < 3)      (retries >= 3)   │
 *      │                               │               │             │
 *      │                        [timer fires]    [still retrying     │
 *      │                               │          in background]     │
 *      │                               ▼               │             │
 *      │                           attempting ◄────────┘             │
 *      │                                                             │
 *      └──────────────[TOOL_CALLS_RESOLVED]──────────────────────────┘
 *                    (success detected via SSE)
 * ```
 *
 * ## Cleanup
 *
 * Requests are cancelled and state is reset when:
 * - YOLO mode is disabled (enabled=false)
 * - Session changes
 * - Component unmounts
 */
export function useAutoApproval({
  enabled,
  sessionId,
  pendingToolCalls,
  pendingToolCallIds,
}: UseAutoApprovalOptions): UseAutoApprovalResult {
  // ─────────────────────────────────────────────────────────────────────────
  // State Machine
  // ─────────────────────────────────────────────────────────────────────────

  const [state, dispatch] = useReducer(
    approvalReducer,
    null,
    createInitialState,
  );

  // Ref mirrors state for synchronous access in guards (before React re-renders)
  const stateRef = useRef(state);

  // Transition helper: updates ref synchronously, then dispatches to React
  const transition = useCallback((action: ApprovalAction) => {
    stateRef.current = approvalReducer(stateRef.current, action);
    dispatch(action);
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Refs
  // ─────────────────────────────────────────────────────────────────────────

  const abortControllerRef = useRef<AbortController | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stable refs to latest values for async callbacks
  const enabledRef = useLatest(enabled);
  const pendingToolCallsRef = useLatest(pendingToolCalls);
  const pendingToolCallIdsRef = useLatest(pendingToolCallIds);

  // ─────────────────────────────────────────────────────────────────────────
  // Cleanup Helpers
  // ─────────────────────────────────────────────────────────────────────────

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const abortAndReset = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    clearRetryTimer();
    transition({ type: "RESET" });
  }, [clearRetryTimer, transition]);

  // ─────────────────────────────────────────────────────────────────────────
  // Core Approval Function
  // ─────────────────────────────────────────────────────────────────────────

  const attemptBatchApproval = useCallback(() => {
    // Synchronous guard using ref (prevents race conditions before React re-renders)
    if (!canStartAttempt(stateRef.current.status)) {
      return;
    }

    const currentPending = pendingToolCallsRef.current;
    if (currentPending.length === 0) {
      return;
    }

    // Approve all pending calls up to the most recent ID
    const lastId = currentPending[currentPending.length - 1].id;

    transition({ type: "START_ATTEMPT", id: lastId });

    // Cancel any pending retry timer (new attempt preempts backoff)
    clearRetryTimer();

    // Ensure we have an AbortController
    if (!abortControllerRef.current) {
      abortControllerRef.current = new AbortController();
    }

    const handleSuccess = () => {
      transition({ type: "ATTEMPT_SUCCESS" });
    };

    const handleFailure = (error: Error) => {
      // Ignore aborted requests
      if (error.name === "AbortError") return;

      // Ignore if disabled while in-flight
      if (!enabledRef.current) return;

      const currentRetryCount = stateRef.current.retryCount;
      const shouldShowError =
        currentRetryCount + 1 >= RETRY_SHOW_ERROR_THRESHOLD;

      if (shouldShowError) {
        // Mark as failed but continue retrying in background
        transition({
          type: "MAX_RETRIES_EXCEEDED",
          failedIds: new Set(pendingToolCallIdsRef.current),
        });
      } else {
        transition({ type: "SCHEDULE_RETRY" });
      }

      // Schedule retry with exponential backoff
      const delay = getRetryDelay(currentRetryCount + 1);
      retryTimerRef.current = setTimeout(() => {
        retryTimerRef.current = null;

        // Guard: could have been disabled during delay
        if (!enabledRef.current) return;
        if (pendingToolCallsRef.current.length === 0) return;

        attemptBatchApproval();
      }, delay);
    };

    approveAllToolCalls(
      sessionId,
      lastId,
      abortControllerRef.current.signal,
    ).then(handleSuccess, handleFailure);
  }, [
    sessionId,
    enabledRef,
    pendingToolCallsRef,
    pendingToolCallIdsRef,
    transition,
    clearRetryTimer,
  ]);

  // ─────────────────────────────────────────────────────────────────────────
  // Effects
  // ─────────────────────────────────────────────────────────────────────────

  // Effect: Handle tool calls being resolved (success detected via SSE)
  useEffect(() => {
    transition({ type: "TOOL_CALLS_RESOLVED", pendingIds: pendingToolCallIds });
  }, [pendingToolCallIds, transition]);

  // Effect: Manage lifecycle (enabled/session changes)
  useEffect(() => {
    if (!enabled) {
      abortAndReset();
      return;
    }

    // Create fresh AbortController when enabled
    abortControllerRef.current = new AbortController();

    return () => {
      abortAndReset();
    };
  }, [enabled, sessionId, abortAndReset]);

  // Effect: Trigger approval when new pending tool calls arrive
  useEffect(() => {
    if (!enabled || pendingToolCalls.length === 0) return;

    attemptBatchApproval();
  }, [enabled, pendingToolCalls, attemptBatchApproval]);

  // ─────────────────────────────────────────────────────────────────────────
  // Return
  // ─────────────────────────────────────────────────────────────────────────

  return { failedIds: state.failedIds };
}
