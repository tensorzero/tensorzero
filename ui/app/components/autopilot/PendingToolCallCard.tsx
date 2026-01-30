import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { Button } from "~/components/ui/button";
import { DotSeparator } from "~/components/ui/TableItems";
import type { GatewayEvent } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import {
  getToolCallEventId,
  isToolEvent,
  ToolEventMetadata,
} from "./EventStream";

type PendingToolCallCardProps = {
  event: GatewayEvent;
  isLoading: boolean;
  loadingAction?: "approving" | "rejecting" | "approving_all";
  onApprove: (approved: boolean) => void;
  onApproveAll?: () => void;
  additionalCount: number;
  isInCooldown?: boolean;
  className?: string;
};

export function PendingToolCallCard({
  event,
  isLoading,
  loadingAction,
  onApprove,
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
    onApprove(true);
  };

  const handleRejectClick = () => {
    setConfirmReject(true);
  };

  const handleRejectConfirm = () => {
    onApprove(false);
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
        "border-card-highlight-border bg-card-highlight flex flex-col gap-2 rounded-md border px-4 py-3",
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
              "text-fg-muted inline-flex",
              isExpanded && "rotate-180",
            )}
          >
            <ChevronDown className="h-4 w-4" />
          </span>
          <span className="rounded bg-orange-100 px-1.5 py-0.5 text-xs font-medium text-orange-700 dark:bg-orange-900 dark:text-orange-200">
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
                <span className="flex h-6 items-center rounded bg-orange-100 px-1.5 text-xs font-medium text-orange-700 dark:bg-orange-900 dark:text-orange-200">
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
                variant="outline"
                size="xs"
                disabled={isDisabled}
                onClick={handleRejectClick}
              >
                Reject
              </Button>
            </div>
          )}
          <ToolEventMetadata
            toolCallEventId={getToolCallEventId(event)}
            timestamp={event.created_at}
          />
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
