import { useCallback, useMemo, useRef } from "react";
import { approveAllToolCalls } from "~/utils/autopilot/approve-all";
import { submitToolCallAuthorization } from "~/utils/autopilot/authorize";
import { approvedStatus, rejectedStatus } from "~/utils/autopilot/types";

interface UseManualAuthorizationResult {
  /** Approve a single tool call. Throws on error. */
  approve: (eventId: string) => Promise<void>;

  /** Reject a single tool call. Throws on error. */
  reject: (eventId: string) => Promise<void>;

  /** Approve all pending tool calls. Throws on error. */
  approveAll: (eventIds: string[], lastEventId: string) => Promise<void>;

  /** Check if an eventId has been processed (in-flight or completed). */
  isProcessed: (eventId: string) => boolean;

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
export function useManualAuthorization(
  sessionId: string,
): UseManualAuthorizationResult {
  const processedRef = useRef<Set<string>>(new Set());

  const approve = useCallback(
    async (eventId: string): Promise<void> => {
      if (processedRef.current.has(eventId)) return;
      processedRef.current.add(eventId);

      try {
        await submitToolCallAuthorization(sessionId, eventId, approvedStatus());
      } catch (err) {
        processedRef.current.delete(eventId);
        throw err;
      }
    },
    [sessionId],
  );

  const reject = useCallback(
    async (eventId: string): Promise<void> => {
      if (processedRef.current.has(eventId)) return;
      processedRef.current.add(eventId);

      try {
        await submitToolCallAuthorization(
          sessionId,
          eventId,
          rejectedStatus("The user rejected the tool call."),
        );
      } catch (err) {
        processedRef.current.delete(eventId);
        throw err;
      }
    },
    [sessionId],
  );

  const approveAll = useCallback(
    async (eventIds: string[], lastEventId: string): Promise<void> => {
      // Mark all event IDs as processed to prevent individual approve/reject
      // from firing duplicate requests while batch is in flight
      const newIds = eventIds.filter((id) => !processedRef.current.has(id));
      if (newIds.length === 0) return;

      for (const id of newIds) {
        processedRef.current.add(id);
      }

      try {
        await approveAllToolCalls(sessionId, lastEventId);
      } catch (err) {
        // Clear all on error to allow retry
        for (const id of newIds) {
          processedRef.current.delete(id);
        }
        throw err;
      }
    },
    [sessionId],
  );

  const isProcessed = useCallback((eventId: string): boolean => {
    return processedRef.current.has(eventId);
  }, []);

  const reset = useCallback(() => {
    processedRef.current.clear();
  }, []);

  return useMemo(
    () => ({ approve, reject, approveAll, isProcessed, reset }),
    [approve, reject, approveAll, isProcessed, reset],
  );
}
