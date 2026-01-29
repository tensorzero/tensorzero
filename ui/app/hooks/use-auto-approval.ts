import { useCallback, useEffect, useRef, useState } from "react";

interface UseAutoApprovalOptions {
  /** Whether auto-approval is enabled */
  enabled: boolean;
  /** List of pending tool call IDs to auto-approve */
  pendingToolCallIds: string[];
  /** Function to authorize a tool call (returns promise, throws on error) */
  onAuthorize: (eventId: string) => Promise<void>;
}

interface UseAutoApprovalResult {
  /** Set of tool call IDs that failed auto-approval after max retries */
  failedIds: Set<string>;
  /** Reset all retry state (call on session change) */
  reset: () => void;
}

/**
 * Hook to handle auto-approval of tool calls with exponential backoff retry.
 *
 * Retry schedule: 1s, 2s, 4s, then 60s forever after showing error.
 * After 3 retries, the tool call ID is added to `failedIds`.
 *
 * Race condition handling:
 * - Uses refs to track in-flight requests and current enabled state
 * - Cleans up timers when tool calls are no longer pending
 * - Cleans up all state when disabled or reset
 */
export function useAutoApproval({
  enabled,
  pendingToolCallIds,
  onAuthorize,
}: UseAutoApprovalOptions): UseAutoApprovalResult {
  // Ref to access current enabled state in async callbacks (avoid stale closures)
  const enabledRef = useRef(enabled);
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  // Ref to access current onAuthorize in async callbacks (avoid stale closures)
  const onAuthorizeRef = useRef(onAuthorize);
  useEffect(() => {
    onAuthorizeRef.current = onAuthorize;
  }, [onAuthorize]);

  // Ref to access current pendingToolCallIds in async callbacks
  const pendingToolCallIdsRef = useRef(pendingToolCallIds);
  useEffect(() => {
    pendingToolCallIdsRef.current = pendingToolCallIds;
  }, [pendingToolCallIds]);

  // Retry tracking
  const retryCountsRef = useRef<Map<string, number>>(new Map());
  const retryTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  // Prevent duplicate in-flight requests
  const inFlightRef = useRef<Set<string>>(new Set());

  // Track tool calls that failed after max retries
  const [failedIds, setFailedIds] = useState<Set<string>>(new Set());

  // Reset all state
  const reset = useCallback(() => {
    for (const timer of retryTimersRef.current.values()) {
      clearTimeout(timer);
    }
    retryTimersRef.current.clear();
    retryCountsRef.current.clear();
    inFlightRef.current.clear();
    setFailedIds(new Set());
  }, []);

  // Cleanup when tool calls are no longer pending
  useEffect(() => {
    const pendingSet = new Set(pendingToolCallIds);

    // Cleanup timers for tool calls no longer pending
    for (const [eventId, timer] of retryTimersRef.current) {
      if (!pendingSet.has(eventId)) {
        clearTimeout(timer);
        retryTimersRef.current.delete(eventId);
        retryCountsRef.current.delete(eventId);
      }
    }

    // Cleanup in-flight tracking for tool calls no longer pending
    for (const eventId of inFlightRef.current) {
      if (!pendingSet.has(eventId)) {
        inFlightRef.current.delete(eventId);
        retryCountsRef.current.delete(eventId);
      }
    }

    // Cleanup failed IDs for tool calls no longer pending
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

  // Cleanup all state when disabled
  useEffect(() => {
    if (!enabled) {
      reset();
    }
  }, [enabled, reset]);

  // Calculate retry delay with exponential backoff
  // 1s, 2s, 4s, then 60s forever after showing error
  const getRetryDelay = (retryCount: number): number => {
    if (retryCount < 3) {
      return 1000 * Math.pow(2, retryCount);
    }
    return 60000;
  };

  // Attempt approval with retry scheduling on failure
  const attemptApproval = useCallback((eventId: string) => {
    // Mark as in-flight
    inFlightRef.current.add(eventId);

    onAuthorizeRef.current(eventId).then(
      () => {
        // Success - remove from in-flight
        // Cleanup of other state happens when pendingToolCallIds changes via SSE
        inFlightRef.current.delete(eventId);
      },
      () => {
        // Failure - remove from in-flight
        inFlightRef.current.delete(eventId);

        // Check if still enabled and eventId still pending (guards against session change)
        if (!enabledRef.current) {
          return;
        }
        if (!pendingToolCallIdsRef.current.includes(eventId)) {
          return;
        }

        // Increment retry count
        const retryCount = (retryCountsRef.current.get(eventId) ?? 0) + 1;
        retryCountsRef.current.set(eventId, retryCount);

        // After 3 retries, show error banner
        if (retryCount === 3) {
          setFailedIds((prev) => new Set(prev).add(eventId));
        }

        // Schedule retry - timer callback directly attempts approval
        const delay = getRetryDelay(retryCount);
        const timer = setTimeout(() => {
          retryTimersRef.current.delete(eventId);

          // Check if still enabled and eventId still pending before retrying
          if (!enabledRef.current) {
            return;
          }
          if (!pendingToolCallIdsRef.current.includes(eventId)) {
            return;
          }

          attemptApproval(eventId);
        }, delay);
        retryTimersRef.current.set(eventId, timer);
      },
    );
  }, []);

  // Auto-approval effect - initiates approval for new pending tool calls
  useEffect(() => {
    if (!enabled || pendingToolCallIds.length === 0) {
      return;
    }

    for (const eventId of pendingToolCallIds) {
      // Skip if already in-flight or has active retry timer
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
