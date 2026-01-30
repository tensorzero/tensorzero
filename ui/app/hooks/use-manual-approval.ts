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
  /** Approve or reject a single tool call. */
  approve: (eventId: string, approved: boolean) => Promise<void>;

  /** Approve all pending tool calls up to lastEventId. */
  approveAll: (lastEventId: string) => Promise<void>;

  /** Check if an approval is in flight for a specific eventId. */
  isInFlight: (eventId: string) => boolean;

  /** Check if a batch approval is in flight. */
  isBatchInFlight: () => boolean;

  /** Reset all in-flight tracking (call on session change). */
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
 * - Check ref synchronously before starting request
 * - Update ref synchronously to block concurrent attempts
 * - Keep ref set on success (blocks clicks until SSE removes tool call from UI)
 * - Clear ref only on error (allows retry)
 *
 * ## Usage
 *
 * ```tsx
 * const { approve, approveAll, isInFlight, reset } = useManualApproval({
 *   sessionId,
 *   showError: toast.error,
 * });
 *
 * // In session change effect:
 * useEffect(() => { reset(); }, [sessionId, reset]);
 *
 * // In click handler (guard before setting loading state):
 * const handleApprove = async (eventId: string) => {
 *   if (isInFlight(eventId)) return;
 *   setLoading(eventId, true);
 *   await approve(eventId, true);
 *   setLoading(eventId, false);
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

  // Track processed authorizations - blocks duplicate requests for same eventId.
  // Stays set on success (until session change) to prevent post-completion clicks.
  const processedRef = useRef<Set<string>>(new Set());

  // Track batch approval - blocks concurrent batch requests.
  // Stays set on success (until session change) to prevent post-completion clicks.
  const batchProcessedRef = useRef<boolean>(false);

  // ─────────────────────────────────────────────────────────────────────────
  // Authorization Functions
  // ─────────────────────────────────────────────────────────────────────────

  const approve = useCallback(
    async (eventId: string, approved: boolean): Promise<void> => {
      // Synchronous guard: prevent duplicate requests for same event
      if (processedRef.current.has(eventId)) return;
      processedRef.current.add(eventId);

      try {
        await authorizeToolCall(
          sessionId,
          eventId,
          approved
            ? approvedStatus()
            : rejectedStatus("The user rejected the tool call."),
        );
        // Success: keep eventId in ref permanently to block any further clicks
        // (the tool call will be removed from pending list via SSE)
      } catch (err) {
        logger.error("Failed to approve tool call:", err);
        showError({
          title: "Approval failed",
          description: "Failed to submit tool call approval. Please try again.",
        });
        // Only clear on error so user can retry
        processedRef.current.delete(eventId);
      }
    },
    [sessionId, showError],
  );

  const approveAll = useCallback(
    async (lastEventId: string): Promise<void> => {
      // Synchronous guard: prevent duplicate batch requests
      if (batchProcessedRef.current) return;
      batchProcessedRef.current = true;

      try {
        await approveAllToolCalls(sessionId, lastEventId);
        // Success: keep ref set to block further batch clicks
        // (pending tool calls will be cleared via SSE)
      } catch (err) {
        logger.error("Failed to approve all tool calls:", err);
        showError({
          title: "Batch approval failed",
          description: "Failed to approve all tool calls. Please try again.",
        });
        // Only clear on error so user can retry
        batchProcessedRef.current = false;
      }
    },
    [sessionId, showError],
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Query Functions
  // ─────────────────────────────────────────────────────────────────────────

  const isInFlight = useCallback((eventId: string): boolean => {
    return processedRef.current.has(eventId);
  }, []);

  const isBatchInFlight = useCallback((): boolean => {
    return batchProcessedRef.current;
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Reset
  // ─────────────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    processedRef.current.clear();
    batchProcessedRef.current = false;
  }, []);

  return useMemo(
    () => ({
      approve,
      approveAll,
      isInFlight,
      isBatchInFlight,
      reset,
    }),
    [approve, approveAll, isInFlight, isBatchInFlight, reset],
  );
}
