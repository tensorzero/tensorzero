import { AlertTriangle, ChevronRight } from "lucide-react";
import { type RefObject, useState } from "react";
import { Skeleton } from "~/components/ui/skeleton";
import { TableItemTime } from "~/components/ui/TableItems";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { Event, InputMessageContent } from "~/types/tensorzero";
import { cn } from "~/utils/common";

/**
 * Optimistic messages are shown after the API confirms receipt but before
 * SSE delivers the real event.
 *
 * Flow:
 * 1. User sends message → POST request fires
 * 2. API responds with event_id → optimistic message created (eventId already set)
 * 3. SSE delivers event with matching ID → optimistic message is removed
 *
 * By waiting for the API response, we always have the eventId, eliminating
 * the race condition where SSE could arrive before we know the ID.
 */
export type OptimisticMessage = {
  tempId: string; // Client-generated UUID, only used as React key
  eventId: string; // Real ID from API response, used to match SSE events
  text: string;
  status: "sending" | "failed";
};

type EventSummary = {
  description?: string;
};

type EventStreamProps = {
  events: Event[];
  className?: string;
  isLoadingOlder?: boolean;
  hasReachedStart?: boolean;
  topSentinelRef?: RefObject<HTMLDivElement | null>;
  pendingToolCallIds?: Set<string>;
  authLoadingStates?: Map<string, "approving" | "rejecting">;
  onAuthorize?: (eventId: string, approved: boolean) => Promise<void>;
  optimisticMessages?: OptimisticMessage[];
};

export function ToolEventId({ id }: { id: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-block max-w-12 cursor-help overflow-hidden align-middle font-mono text-xs text-ellipsis whitespace-nowrap"
          dir="rtl"
        >
          {id}
        </span>
      </TooltipTrigger>
      <TooltipContent
        className="border-border bg-bg-secondary text-fg-primary border text-xs shadow-lg"
        sideOffset={5}
      >
        Tool Call ID: <span className="font-mono text-xs">{id}</span>
      </TooltipContent>
    </Tooltip>
  );
}

function getMessageText(content: InputMessageContent[]) {
  const textBlock = content.find(
    (block) => block.type === "text" && "text" in block,
  );
  if (textBlock && "text" in textBlock) {
    return textBlock.text;
  }

  const rawTextBlock = content.find(
    (block) => block.type === "raw_text" && "value" in block,
  );
  if (rawTextBlock && "value" in rawTextBlock) {
    return rawTextBlock.value;
  }

  return "Message content";
}

function summarizeEvent(event: Event): EventSummary {
  const { payload } = event;

  switch (payload.type) {
    case "message":
      return {
        description: getMessageText(payload.content),
      };
    case "status_update":
      return {
        description: payload.status_update.text,
      };
    case "tool_call":
      return {
        description: payload.arguments,
      };
    case "tool_call_authorization":
      return {
        description:
          payload.status.type === "rejected"
            ? payload.status.reason
            : undefined,
      };
    case "tool_result":
      if (payload.outcome.type === "success") {
        return {
          description: payload.outcome.result,
        };
      }
      if (payload.outcome.type === "failure") {
        return {
          description: payload.outcome.message,
        };
      }
      return {};
    case "other":
      return {};
    default:
      return {};
  }
}

function renderEventTitle(event: Event) {
  const { payload } = event;

  switch (payload.type) {
    case "message": {
      const roleLabel =
        payload.role === "user"
          ? "User"
          : payload.role === "assistant"
            ? "Assistant"
            : "Message";
      return `${roleLabel} Message`;
    }
    case "status_update":
      return "Status Update";
    case "tool_call":
      return (
        <>
          Tool Call &middot;{" "}
          <span className="font-mono font-medium">{payload.name}</span>
        </>
      );
    case "tool_call_authorization":
      switch (payload.status.type) {
        case "approved":
          return <>Tool Call Authorization &middot; Approved</>;
        case "rejected":
          return <>Tool Call Authorization &middot; Rejected</>;
      }
      break;
    case "tool_result":
      switch (payload.outcome.type) {
        case "success":
          return (
            <>
              Tool Result &middot;{" "}
              <span className="font-mono font-medium">
                {payload.outcome.name}
              </span>
            </>
          );
        case "failure":
          // TODO: need tool name
          return <>Tool Result &middot; Failure</>;
        case "missing":
          // TODO: need tool name
          return (
            <span className="inline-flex items-center gap-2">
              <span>Tool Result &middot; Missing Tool</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="inline-flex cursor-help items-center text-yellow-600"
                    aria-label="Missing tool warning"
                  >
                    <AlertTriangle className="h-4 w-4" />
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  The agent requested a tool that your gateway does not support.
                </TooltipContent>
              </Tooltip>
            </span>
          );
        case "other":
          // TODO: need tool name
          return (
            <span className="inline-flex items-center gap-2">
              <span>Tool Result &middot; Unknown</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="inline-flex cursor-help items-center text-yellow-600"
                    aria-label="Unknown tool result"
                  >
                    <AlertTriangle className="h-4 w-4" />
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  The Autopilot API returned an unknown event. This likely means
                  your TensorZero deployment version is outdated.
                </TooltipContent>
              </Tooltip>
            </span>
          );
      }
      break;
    case "other":
      return (
        <span className="inline-flex items-center gap-2">
          <span>Unknown Event</span>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className="inline-flex cursor-help items-center text-yellow-600"
                aria-label="Unknown event"
              >
                <AlertTriangle className="h-4 w-4" />
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              The Autopilot API returned an unknown event. This likely means
              your TensorZero deployment version is outdated.
            </TooltipContent>
          </Tooltip>
        </span>
      );
    default:
      return "Event";
  }
}

function EventItem({
  event,
  isPending = false,
}: {
  event: Event;
  isPending?: boolean;
}) {
  const summary = summarizeEvent(event);
  const title = renderEventTitle(event);
  const isToolEvent =
    event.payload.type === "tool_call" ||
    event.payload.type === "tool_call_authorization" ||
    event.payload.type === "tool_result";
  const isExpandable =
    event.payload.type === "tool_call" ||
    (event.payload.type === "tool_call_authorization" &&
      event.payload.status.type === "rejected") ||
    (event.payload.type === "tool_result" &&
      (event.payload.outcome.type === "success" ||
        event.payload.outcome.type === "failure"));
  const [isExpanded, setIsExpanded] = useState(false);
  const shouldShowDetails = !isExpandable || isExpanded;
  const label = <span className="text-sm font-medium">{title}</span>;

  return (
    <div className="border-border bg-bg-secondary flex flex-col gap-2 rounded-md border px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        {isExpandable ? (
          <button
            type="button"
            aria-expanded={isExpanded}
            aria-label={
              isExpanded ? "Collapse tool details" : "Expand tool details"
            }
            className="inline-flex cursor-pointer items-center gap-2 text-left"
            onClick={() => setIsExpanded((current) => !current)}
          >
            {label}
            <span
              className={cn(
                "text-fg-muted inline-flex transition-transform duration-200",
                isExpanded ? "rotate-90" : "rotate-0",
              )}
            >
              <ChevronRight className="h-4 w-4" />
            </span>
            {isPending && (
              <span className="rounded bg-blue-200 px-1.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
                Action Required
              </span>
            )}
          </button>
        ) : (
          label
        )}
        <div className="text-fg-muted flex items-center gap-1.5 text-xs">
          {isToolEvent && (
            <>
              <ToolEventId id={event.id} />
              <span aria-hidden="true">&middot;</span>
            </>
          )}
          <TableItemTime timestamp={event.created_at} />
        </div>
      </div>
      {shouldShowDetails && summary.description && (
        <p
          className={cn(
            "text-fg-secondary whitespace-pre-wrap",
            event.payload.type === "tool_call" ||
              event.payload.type === "tool_result"
              ? "font-mono text-sm"
              : "text-sm",
          )}
        >
          {summary.description}
        </p>
      )}
    </div>
  );
}

function EventSkeletons({ count = 3 }: { count?: number }) {
  return (
    <div className="flex flex-col gap-3">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="border-border bg-bg-secondary rounded-md border px-4 py-3"
        >
          <div className="flex items-center justify-between gap-4">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-4 w-24" />
          </div>
          <Skeleton className="mt-2 h-4 w-3/4" />
        </div>
      ))}
    </div>
  );
}

function SessionStartedDivider() {
  return (
    <div className="flex items-center gap-4 py-2">
      <div className="border-border flex-1 border-t" />
      <span className="text-fg-muted text-xs">Session Started</span>
      <div className="border-border flex-1 border-t" />
    </div>
  );
}

function OptimisticMessageItem({ message }: { message: OptimisticMessage }) {
  // Optimistic messages are shown after POST succeeds, so the message is saved.
  // We display it like a regular user message, with a skeleton for the timestamp.
  return (
    <div className="border-border bg-bg-secondary flex flex-col gap-2 rounded-md border px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        <span className="text-sm font-medium">User Message</span>
        <Skeleton className="h-4 w-32" />
      </div>
      <p className="text-fg-secondary text-sm whitespace-pre-wrap">
        {message.text}
      </p>
    </div>
  );
}

export default function EventStream({
  events,
  className,
  isLoadingOlder = false,
  hasReachedStart = false,
  topSentinelRef,
  pendingToolCallIds,
  optimisticMessages = [],
}: EventStreamProps) {
  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* Session started indicator, or sentinel for loading more */}
      {/* Show divider when we've reached the start OR when there are optimistic messages (new session) */}
      {(hasReachedStart || optimisticMessages.length > 0) && !isLoadingOlder ? (
        <SessionStartedDivider />
      ) : (
        <div ref={topSentinelRef} className="h-1" aria-hidden="true" />
      )}

      {/* Loading skeletons at the top */}
      {isLoadingOlder && <EventSkeletons count={3} />}

      {events.map((event) => (
        <EventItem
          key={event.id}
          event={event}
          isPending={pendingToolCallIds?.has(event.id)}
        />
      ))}

      {/* Optimistic messages at the end */}
      {optimisticMessages.map((message) => (
        <OptimisticMessageItem key={message.tempId} message={message} />
      ))}
    </div>
  );
}
