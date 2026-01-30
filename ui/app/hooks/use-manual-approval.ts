import { useCallback, useMemo, useRef } from "react";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { authorizeToolCall } from "~/utils/autopilot/authorize";
import { approvedStatus, rejectedStatus } from "~/utils/autopilot/types";

interface UseManualApprovalResult {
  /** Approve or reject a single tool call. Throws on error. */
  approve: (eventId: string, approved: boolean) => Promise<void>;

  /** Approve all pending tool calls. Throws on error. */
  approveAll: (lastEventId: string) => Promise<void>;

  /** Check if an eventId has been processed (in-flight or completed). */
  isProcessed: (eventId: string) => boolean;

  /** Check if batch approval has been processed. */
  isBatchProcessed: () => boolean;

  /** Reset all tracking (call on session change). */
  reset: () => void;
}

/**
 * Handle manual tool call authorization with deduplication.
 *
 * Uses synchronous ref-based guards to prevent duplicate API calls when user
 * spam-clicks approve/reject buttons.
 *
 * ## Pattern
 *
 * - Check ref synchronously before starting request
 * - Update ref synchronously to block concurrent attempts
 * - Keep ref set on success (blocks until SSE removes tool call from UI)
 * - Clear ref only on error (allows retry)
 */
export function useManualApproval(sessionId: string): UseManualApprovalResult {
  const processedRef = useRef<Set<string>>(new Set());
  const batchProcessedRef = useRef<boolean>(false);

  const approve = useCallback(
    async (eventId: string, approved: boolean): Promise<void> => {
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
      } catch (err) {
        processedRef.current.delete(eventId);
        throw err;
      }
    },
    [sessionId],
  );

  const approveAll = useCallback(
    async (lastEventId: string): Promise<void> => {
      if (batchProcessedRef.current) return;
      batchProcessedRef.current = true;

      try {
        await approveAllToolCalls(sessionId, lastEventId);
      } catch (err) {
        batchProcessedRef.current = false;
        throw err;
      }
    },
    [sessionId],
  );

  const isProcessed = useCallback((eventId: string): boolean => {
    return processedRef.current.has(eventId);
  }, []);

  const isBatchProcessed = useCallback((): boolean => {
    return batchProcessedRef.current;
  }, []);

  const reset = useCallback(() => {
    processedRef.current.clear();
    batchProcessedRef.current = false;
  }, []);

  return useMemo(
    () => ({ approve, approveAll, isProcessed, isBatchProcessed, reset }),
    [approve, approveAll, isProcessed, isBatchProcessed, reset],
  );
}
