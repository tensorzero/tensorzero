import { useCallback, useEffect, useRef, useState } from "react";
import { useLocalStorage } from "~/hooks/use-local-storage";
import { useToast } from "~/hooks/use-toast";
import { logger } from "~/utils/logger";
import type { GatewayEvent } from "~/types/tensorzero";

interface UseYoloModeOptions {
  sessionId: string;
  pendingToolCalls: GatewayEvent[];
  onApproveToolCall: (eventId: string) => Promise<void>;
  onApproveAll: (lastToolCallEventId: string) => Promise<void>;
}

interface UseYoloModeResult {
  isYoloEnabled: boolean;
  setYoloEnabled: (enabled: boolean) => void;
  currentAutoApprovingTool: string | null;
  autoApprovedCount: number;
}

/**
 * Hook that manages yolo mode state and auto-approval logic.
 *
 * When yolo mode is enabled:
 * 1. Clears all pending tool calls via approveAll
 * 2. Auto-approves new tool calls as they arrive via SSE
 *
 * State is persisted in localStorage with key "autopilot-yolo-mode".
 */
export function useYoloMode({
  sessionId,
  pendingToolCalls,
  onApproveToolCall,
  onApproveAll,
}: UseYoloModeOptions): UseYoloModeResult {
  const [isYoloEnabled, setYoloEnabledRaw] = useLocalStorage<boolean>(
    "autopilot-yolo-mode",
    false,
  );
  const [currentAutoApprovingTool, setCurrentAutoApprovingTool] = useState<
    string | null
  >(null);
  const [autoApprovedCount, setAutoApprovedCount] = useState(0);
  const { toast } = useToast();

  // Track which tool calls we've already tried to auto-approve
  const processedToolCallsRef = useRef<Set<string>>(new Set());

  // Track if we're currently processing to avoid race conditions
  const isProcessingRef = useRef(false);

  // Reset processed tool calls when session changes
  useEffect(() => {
    processedToolCallsRef.current = new Set();
    setAutoApprovedCount(0);
    setCurrentAutoApprovingTool(null);
  }, [sessionId]);

  // Handle enabling yolo mode - clear queue first
  const setYoloEnabled = useCallback(
    async (enabled: boolean) => {
      setYoloEnabledRaw(enabled);

      if (enabled && pendingToolCalls.length > 0) {
        // Mark all current tool calls as processed (they'll be handled by approveAll)
        for (const event of pendingToolCalls) {
          processedToolCallsRef.current.add(event.id);
        }

        // Get the last tool call ID for approveAll
        const lastToolCallId = pendingToolCalls[pendingToolCalls.length - 1].id;

        try {
          await onApproveAll(lastToolCallId);
          setAutoApprovedCount((prev) => prev + pendingToolCalls.length);
        } catch (err) {
          logger.error("Failed to approve all tool calls:", err);
          toast.error({
            title: "Failed to approve all",
            description:
              err instanceof Error ? err.message : "Please try again.",
          });
          // Continue with yolo mode even if initial clear fails
        }
      }
    },
    [pendingToolCalls, onApproveAll, setYoloEnabledRaw, toast],
  );

  // Auto-approve new tool calls when yolo mode is enabled
  useEffect(() => {
    if (
      !isYoloEnabled ||
      pendingToolCalls.length === 0 ||
      isProcessingRef.current
    ) {
      return;
    }

    // Find tool calls that haven't been processed yet
    const unprocessedToolCalls = pendingToolCalls.filter(
      (event) => !processedToolCallsRef.current.has(event.id),
    );

    if (unprocessedToolCalls.length === 0) {
      return;
    }

    // Process them one at a time
    const processNext = async () => {
      if (isProcessingRef.current) return;

      const nextToolCall = unprocessedToolCalls[0];
      if (!nextToolCall) return;

      // Mark as processing
      isProcessingRef.current = true;
      processedToolCallsRef.current.add(nextToolCall.id);

      // Extract tool name for the indicator
      const toolName =
        nextToolCall.payload.type === "tool_call"
          ? nextToolCall.payload.name
          : "unknown";

      setCurrentAutoApprovingTool(toolName);

      try {
        await onApproveToolCall(nextToolCall.id);
        setAutoApprovedCount((prev) => prev + 1);
      } catch (err) {
        logger.error("Failed to auto-approve tool call:", err);
        toast.error({
          title: "Auto-approve failed",
          description:
            err instanceof Error ? err.message : "Continuing to try...",
        });
        // Keep yolo mode enabled and continue trying
      } finally {
        setCurrentAutoApprovingTool(null);
        isProcessingRef.current = false;
      }
    };

    processNext();
  }, [isYoloEnabled, pendingToolCalls, onApproveToolCall, toast]);

  return {
    isYoloEnabled,
    setYoloEnabled,
    currentAutoApprovingTool,
    autoApprovedCount,
  };
}
