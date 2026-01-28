import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { TableItemTime } from "~/components/ui/TableItems";
import type { GatewayEvent } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { getToolCallEventId, isToolEvent, ToolEventId } from "./EventStream";
import { YoloToggle } from "./YoloModeIndicator";

type PendingToolCallCardProps = {
  event: GatewayEvent;
  isLoading: boolean;
  loadingAction?: "approving" | "rejecting";
  onAuthorize: (approved: boolean) => void;
  additionalCount: number;
  isInCooldown?: boolean;
  // Approve All props
  onApproveAll?: () => void;
  isApproveAllLoading?: boolean;
  // Yolo mode props
  isYoloEnabled?: boolean;
  onYoloToggle?: (enabled: boolean) => void;
};

export function PendingToolCallCard({
  event,
  isLoading,
  loadingAction,
  onAuthorize,
  additionalCount,
  isInCooldown = false,
  onApproveAll,
  isApproveAllLoading = false,
  isYoloEnabled = false,
  onYoloToggle,
}: PendingToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [confirmReject, setConfirmReject] = useState(false);
  const [confirmApproveAll, setConfirmApproveAll] = useState(false);

  // Only handle tool_call events
  if (!isToolEvent(event) || event.payload.type !== "tool_call") {
    return null;
  }

  const { name, arguments: args } = event.payload;

  const handleApprove = () => {
    onAuthorize(true);
  };

  const handleRejectClick = () => {
    setConfirmReject(true);
  };

  const handleRejectConfirm = () => {
    onAuthorize(false);
    setConfirmReject(false);
  };

  const handleRejectCancel = () => {
    setConfirmReject(false);
  };

  const handleApproveAllClick = () => {
    setConfirmApproveAll(true);
  };

  const handleApproveAllConfirm = () => {
    onApproveAll?.();
    setConfirmApproveAll(false);
  };

  const handleApproveAllCancel = () => {
    setConfirmApproveAll(false);
  };

  const isDisabled = isLoading || isInCooldown || isApproveAllLoading;
  const totalPendingCount = additionalCount + 1;

  // When isInCooldown is true, the card animates in and buttons are disabled.
  // This prevents accidental clicks when the queue changes via SSE.
  return (
    <div
      key={event.id}
      className={cn(
        "flex flex-col gap-2 rounded-md border border-blue-300 bg-blue-50 px-4 py-3 dark:border-blue-700 dark:bg-blue-950/30",
        isInCooldown &&
          "animate-in fade-in zoom-in-95 duration-1000 ease-in-out",
      )}
    >
      <div className="flex items-center justify-between gap-4">
        <button
          type="button"
          aria-expanded={isExpanded}
          aria-label={
            isExpanded ? "Collapse tool details" : "Expand tool details"
          }
          className="inline-flex cursor-pointer items-center gap-2 text-left"
          onClick={() => setIsExpanded((current) => !current)}
        >
          <span className="text-sm font-medium">
            Tool Call &middot;{" "}
            <span className="font-mono font-medium">{name}</span>
          </span>
          <span
            className={cn(
              "text-fg-muted inline-flex transition-transform duration-200",
              isExpanded ? "rotate-90" : "rotate-0",
            )}
          >
            <ChevronRight className="h-4 w-4" />
          </span>
          <span className="rounded bg-blue-200 px-1.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
            Action Required
          </span>
        </button>
        <div className="flex items-center gap-3">
          {/* Action buttons inline */}
          {confirmReject ? (
            <div
              className="flex gap-2"
              role="group"
              aria-label="Confirm rejection"
            >
              <button
                type="button"
                className="h-6 cursor-pointer rounded bg-red-600 px-2 text-xs font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isDisabled}
                onClick={handleRejectConfirm}
                aria-label="Confirm rejection"
              >
                {loadingAction === "rejecting" ? "Rejecting..." : "Yes, reject"}
              </button>
              <button
                type="button"
                className="h-6 cursor-pointer rounded bg-gray-100 px-2 text-xs font-medium hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-800 dark:hover:bg-gray-700"
                disabled={isDisabled}
                onClick={handleRejectCancel}
                aria-label="Cancel rejection"
              >
                No, keep it
              </button>
            </div>
          ) : confirmApproveAll ? (
            <div
              className="flex gap-2"
              role="group"
              aria-label="Confirm approve all"
            >
              <button
                type="button"
                className="bg-fg-primary text-bg-primary hover:bg-fg-secondary h-6 cursor-pointer rounded px-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isDisabled}
                onClick={handleApproveAllConfirm}
                aria-label="Confirm approve all"
              >
                {isApproveAllLoading
                  ? "Approving..."
                  : `Approve ${totalPendingCount}?`}
              </button>
              <button
                type="button"
                className="h-6 cursor-pointer rounded bg-gray-100 px-2 text-xs font-medium hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-800 dark:hover:bg-gray-700"
                disabled={isDisabled}
                onClick={handleApproveAllCancel}
                aria-label="Cancel approve all"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              {additionalCount > 0 && (
                <span className="flex h-6 items-center rounded bg-blue-200 px-1.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
                  +{additionalCount}
                </span>
              )}
              {additionalCount > 0 && onApproveAll && (
                <button
                  type="button"
                  className="h-6 cursor-pointer rounded border border-blue-400 px-2 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-600 dark:text-blue-300 dark:hover:bg-blue-900/30"
                  disabled={isDisabled}
                  onClick={handleApproveAllClick}
                  aria-label={`Approve all ${totalPendingCount} pending tool calls`}
                >
                  Approve All
                </button>
              )}
              <button
                type="button"
                className="bg-fg-primary text-bg-primary hover:bg-fg-secondary h-6 cursor-pointer rounded px-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isDisabled}
                onClick={handleApprove}
              >
                {loadingAction === "approving" ? "Approving..." : "Approve"}
              </button>
              <button
                type="button"
                className="h-6 cursor-pointer rounded border border-red-300 px-2 text-xs font-medium text-red-600 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-950/30"
                disabled={isDisabled}
                onClick={handleRejectClick}
              >
                Reject
              </button>
            </div>
          )}
          <div className="text-fg-muted flex items-center gap-1.5 text-xs">
            <ToolEventId id={getToolCallEventId(event)} />
            <span aria-hidden="true">&middot;</span>
            <TableItemTime timestamp={event.created_at} />
          </div>
        </div>
      </div>

      {/* Tool arguments (expandable) */}
      {isExpanded && args && (
        <pre className="text-fg-secondary overflow-x-auto font-mono text-xs whitespace-pre-wrap">
          {JSON.stringify(args, null, 2)}
        </pre>
      )}

      {/* Yolo toggle - always visible at bottom */}
      {onYoloToggle && (
        <div className="flex justify-end border-t border-blue-200 pt-2 dark:border-blue-800">
          <YoloToggle isEnabled={isYoloEnabled} onToggle={onYoloToggle} />
        </div>
      )}
    </div>
  );
}
