import { useCallback, useMemo, useRef } from "react";
import type { ToastErrorProps } from "~/context/toast-context";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { authorizeToolCall } from "~/utils/autopilot/authorize";
import { approvedStatus, rejectedStatus } from "~/utils/autopilot/types";
import { logger } from "~/utils/logger";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface UseManualApprovalOptions {
  sessionId: string;
  /** Called to display error notifications (e.g., toast.error) */
  showError: (error: ToastErrorProps) => void;
}

interface UseManualApprovalResult {
  /**
   * Authorize a single tool call. Returns false if request was deduplicated
   * (already in flight for this eventId).
   */
  authorize: (
    eventId: string,
    approved: boolean,
  ) => Promise<{ success: boolean; deduplicated: boolean }>;

  /**
   * Approve all pending tool calls up to lastEventId. Returns false if request
   * was deduplicated (batch approval already in flight).
   */
  approveAll: (
    lastEventId: string,
  ) => Promise<{ success: boolean; deduplicated: boolean }>;

  /**
   * Check if an authorization is in flight for a specific eventId.
   */
  isInFlight: (eventId: string) => boolean;

  /**
   * Check if a batch approval is in flight.
   */
  isBatchInFlight: () => boolean;

  /**
   * Reset all in-flight tracking (call on session change).
   */
  reset: () => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle manual tool call authorization with deduplication.
 *
 * Uses synchronous ref-based guards to prevent duplicate API calls when user
 * spam-clicks approve/reject buttons. The guards work before React re-renders,
 * closing the race window between click and loading state update.
 *
 * ## Pattern
 *
 * This follows the same synchronous ref pattern as `useAutoApproval`:
 * - Check ref synchronously before starting request
 * - Update ref synchronously to block concurrent attempts
 * - Clear ref in finally block after request completes
 *
 * ## Usage
 *
 * ```tsx
 * const { authorize, approveAll, reset } = useManualApproval({
 *   sessionId,
 *   showError: toast.error,
 * });
 *
 * // In session change effect:
 * useEffect(() => { reset(); }, [sessionId, reset]);
 *
 * // In click handler:
 * const handleApprove = async (eventId: string) => {
 *   setLoading(eventId, true);
 *   const { success, deduplicated } = await authorize(eventId, true);
 *   if (!deduplicated) setLoading(eventId, false);
 * };
 * ```
 */
export function useManualApproval({
  sessionId,
  showError,
}: UseManualApprovalOptions): UseManualApprovalResult {
  // ─────────────────────────────────────────────────────────────────────────
  // Refs for synchronous deduplication
  // ─────────────────────────────────────────────────────────────────────────

  // Track individual authorizations in flight (by eventId)
  const inFlightAuthRef = useRef<Set<string>>(new Set());

  // Track batch approval in flight (by lastEventId to allow new batches)
  const batchInFlightRef = useRef<string | null>(null);

  // ─────────────────────────────────────────────────────────────────────────
  // Authorization Functions
  // ─────────────────────────────────────────────────────────────────────────

  const authorize = useCallback(
    async (
      eventId: string,
      approved: boolean,
    ): Promise<{ success: boolean; deduplicated: boolean }> => {
      // Synchronous guard: prevent duplicate requests for same event
      if (inFlightAuthRef.current.has(eventId)) {
        return { success: false, deduplicated: true };
      }
      inFlightAuthRef.current.add(eventId);

      try {
        await authorizeToolCall(
          sessionId,
          eventId,
          approved
            ? approvedStatus()
            : rejectedStatus("The user rejected the tool call."),
        );
        return { success: true, deduplicated: false };
      } catch (err) {
        logger.error("Failed to authorize tool call:", err);
        showError({
          title: "Authorization failed",
          description:
            "Failed to submit tool call authorization. Please try again.",
        });
        return { success: false, deduplicated: false };
      } finally {
        inFlightAuthRef.current.delete(eventId);
      }
    },
    [sessionId, showError],
  );

  const approveAll = useCallback(
    async (
      lastEventId: string,
    ): Promise<{ success: boolean; deduplicated: boolean }> => {
      // Synchronous guard: prevent duplicate batch requests
      // We track by lastEventId so a new batch with different ID can proceed
      if (batchInFlightRef.current !== null) {
        return { success: false, deduplicated: true };
      }
      batchInFlightRef.current = lastEventId;

      try {
        await approveAllToolCalls(sessionId, lastEventId);
        return { success: true, deduplicated: false };
      } catch (err) {
        logger.error("Failed to approve all tool calls:", err);
        showError({
          title: "Batch approval failed",
          description: "Failed to approve all tool calls. Please try again.",
        });
        return { success: false, deduplicated: false };
      } finally {
        batchInFlightRef.current = null;
      }
    },
    [sessionId, showError],
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Query Functions
  // ─────────────────────────────────────────────────────────────────────────

  const isInFlight = useCallback((eventId: string): boolean => {
    return inFlightAuthRef.current.has(eventId);
  }, []);

  const isBatchInFlight = useCallback((): boolean => {
    return batchInFlightRef.current !== null;
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Reset
  // ─────────────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    inFlightAuthRef.current.clear();
    batchInFlightRef.current = null;
  }, []);

  return useMemo(
    () => ({
      authorize,
      approveAll,
      isInFlight,
      isBatchInFlight,
      reset,
    }),
    [authorize, approveAll, isInFlight, isBatchInFlight, reset],
  );
}
