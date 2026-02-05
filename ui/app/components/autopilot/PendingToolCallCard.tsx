import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { Button } from "~/components/ui/button";
import { DotSeparator } from "~/components/ui/DotSeparator";
import { TableItemTime } from "~/components/ui/TableItems";
import type { GatewayEvent } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { getToolCallEventId, isToolEvent, ToolEventId } from "./EventStream";

type PendingToolCallCardProps = {
  event: GatewayEvent;
  isLoading: boolean;
  loadingAction?: "approving" | "rejecting" | "approving_all";
  onAuthorize: (approved: boolean) => void;
  onApproveAll?: () => void;
  additionalCount: number;
  isInCooldown?: boolean;
  className?: string;
};

export function PendingToolCallCard({
  event,
  isLoading,
  loadingAction,
  onAuthorize,
  onApproveAll,
  additionalCount,
  isInCooldown = false,
  className,
}: PendingToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [confirmReject, setConfirmReject] = useState(false);

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
        className,
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
          <span className="inline-flex items-center gap-2 text-sm font-medium">
            Tool Call
            <DotSeparator />
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
              <Button
                variant="destructive"
                size="xs"
                disabled={isDisabled}
                onClick={handleRejectConfirm}
                aria-label="Confirm rejection"
              >
                {loadingAction === "rejecting" ? "Rejecting..." : "Yes, reject"}
              </Button>
              <Button
                variant="secondary"
                size="xs"
                disabled={isDisabled}
                onClick={handleRejectCancel}
                aria-label="Cancel rejection"
              >
                No, keep it
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              {additionalCount > 0 && (
                <span className="flex h-6 items-center rounded bg-blue-200 px-1.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
                  +{additionalCount}
                </span>
              )}
              {additionalCount > 0 && onApproveAll && (
                <Button
                  variant="successOutline"
                  size="xs"
                  disabled={isDisabled}
                  onClick={onApproveAll}
                >
                  {loadingAction === "approving_all"
                    ? "Approving..."
                    : `Approve All (${additionalCount + 1})`}
                </Button>
              )}
              <Button size="xs" disabled={isDisabled} onClick={handleApprove}>
                {loadingAction === "approving" ? "Approving..." : "Approve"}
              </Button>
              <Button
                variant="destructiveOutline"
                size="xs"
                disabled={isDisabled}
                onClick={handleRejectClick}
              >
                Reject
              </Button>
            </div>
          )}
          <div className="text-fg-muted flex items-center gap-1.5 text-xs">
            <ToolEventId id={getToolCallEventId(event)} />
            <DotSeparator />
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
    </div>
  );
}
