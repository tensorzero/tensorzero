import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { TableItemTime } from "~/components/ui/TableItems";
import type { Event } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { ToolEventId } from "./EventStream";

type PendingToolCallCardProps = {
  event: Event;
  isLoading: boolean;
  loadingAction?: "approving" | "rejecting";
  onAuthorize: (approved: boolean) => void;
  additionalCount: number;
  isInCooldown?: boolean;
};

export function PendingToolCallCard({
  event,
  isLoading,
  loadingAction,
  onAuthorize,
  additionalCount,
  isInCooldown = false,
}: PendingToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [confirmReject, setConfirmReject] = useState(false);

  // Only handle tool_call events
  if (event.payload.type !== "tool_call") {
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

  const isDisabled = isLoading || isInCooldown;

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
          ) : (
            <div className="flex items-center gap-2">
              {additionalCount > 0 && (
                <span className="flex h-6 items-center rounded bg-blue-200 px-1.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
                  +{additionalCount}
                </span>
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
            <ToolEventId id={event.id} />
            <span aria-hidden="true">&middot;</span>
            <TableItemTime timestamp={event.created_at} />
          </div>
        </div>
      </div>

      {/* Tool arguments (expandable) */}
      {isExpanded && args && (
        <p className="text-fg-secondary font-mono text-xs">{args}</p>
      )}
    </div>
  );
}
