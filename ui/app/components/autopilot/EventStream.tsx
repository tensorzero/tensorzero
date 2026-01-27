import { AlertTriangle, ChevronRight, Loader2 } from "lucide-react";
import { type RefObject, useState } from "react";
import { Markdown } from "~/components/ui/markdown";
import { Skeleton } from "~/components/ui/skeleton";
import { TableItemTime } from "~/components/ui/TableItems";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type {
  AutopilotStatus,
  EventPayloadMessageContent,
  GatewayEvent,
  GatewayEventPayload,
} from "~/types/tensorzero";
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
  events: GatewayEvent[];
  className?: string;
  isLoadingOlder?: boolean;
  hasReachedStart?: boolean;
  topSentinelRef?: RefObject<HTMLDivElement | null>;
  pendingToolCallIds?: Set<string>;
  authLoadingStates?: Map<string, "approving" | "rejecting">;
  onAuthorize?: (eventId: string, approved: boolean) => Promise<void>;
  optimisticMessages?: OptimisticMessage[];
  status?: AutopilotStatus;
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

/**
 * Tool event payload types - events related to tool calls.
 */
export type ToolEventPayload = Extract<
  GatewayEventPayload,
  { type: "tool_call" | "tool_call_authorization" | "tool_result" }
>;

/**
 * An event with a tool-related payload.
 */
export type ToolEvent = GatewayEvent & { payload: ToolEventPayload };

/**
 * Type guard to check if an event is a tool event.
 */
export function isToolEvent(event: GatewayEvent): event is ToolEvent {
  return (
    event.payload.type === "tool_call" ||
    event.payload.type === "tool_call_authorization" ||
    event.payload.type === "tool_result"
  );
}

/**
 * Extracts the tool_call_event_id from a tool event.
 * For tool_call events, this is in side_info.tool_call_event_id.
 * For tool_call_authorization and tool_result events, this is directly on the payload.
 */
export function getToolCallEventId(event: ToolEvent): string {
  const { payload } = event;
  if (payload.type === "tool_call") {
    return payload.side_info.tool_call_event_id;
  }
  return payload.tool_call_event_id;
}

function getMessageText(content: EventPayloadMessageContent[]) {
  return content.map((cb) => cb.text).join("\n\n");
}

/**
 * Format a tool error for display.
 * Extracts a human-readable message from the structured error JSON.
 */
function formatToolError(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const e = error as Record<string, unknown>;
    // AutopilotToolError has a "message" field
    if (typeof e.message === "string") {
      return e.message;
    }
  }
  // Fallback to JSON stringification
  return JSON.stringify(error);
}

function summarizeEvent(event: GatewayEvent): EventSummary {
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
        description: JSON.stringify(payload.arguments, null, 2),
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
          description: JSON.stringify(payload.outcome.result, null, 2),
        };
      }
      if (payload.outcome.type === "failure") {
        return {
          description: formatToolError(payload.outcome.error),
        };
      }
      return {};
    case "error":
      // TODO: handle errors
      return {};
    case "unknown":
      return {};
    default:
      return {};
  }
}

function renderEventTitle(event: GatewayEvent) {
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
        default:
          // This branch should never be reached but we need it to keep ESLint happy...
          {
            const _exhaustiveCheck: never = payload.status; // TS compiler should yell if this branch is reachable
          }
          throw new Error(
            "Unknown tool call authorization status. This should never happen. Please open a bug report: https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
          );
      }
    case "tool_result":
      switch (payload.outcome.type) {
        case "success":
          // TODO: need tool name
          return <>Tool Result &middot; Success</>;
        case "failure":
          // TODO: need tool name
          return <>Tool Result &middot; Failure</>;
        case "rejected":
          // TODO: need tool name
          return (
            <span className="inline-flex items-center gap-2">
              <span>Tool Result &middot; Rejected</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className="inline-flex cursor-help items-center text-yellow-600"
                    aria-label="Tool rejected"
                  >
                    <AlertTriangle className="h-4 w-4" />
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  {payload.outcome.reason}
                </TooltipContent>
              </Tooltip>
            </span>
          );
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
        case "unknown":
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
        default:
          // This branch should never be reached but we need it to keep ESLint happy...
          {
            const _exhaustiveCheck: never = payload.outcome; // TS compiler should yell if this branch is reachable
          }
          throw new Error(
            "Unknown tool call authorization status. This should never happen. Please open a bug report: https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
          );
      }
    case "error":
      // TODO: handle errors better
      return "Error";
    case "unknown":
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
  event: GatewayEvent;
  isPending?: boolean;
}) {
  const summary = summarizeEvent(event);
  const title = renderEventTitle(event);
  const eventIsToolEvent = isToolEvent(event);
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
          {eventIsToolEvent && (
            <>
              <ToolEventId id={getToolCallEventId(event)} />
              <span aria-hidden="true">&middot;</span>
            </>
          )}
          <TableItemTime timestamp={event.created_at} />
        </div>
      </div>
      {shouldShowDetails && summary.description && (
        <>
          {event.payload.type === "message" &&
          event.payload.role === "assistant" ? (
            <Markdown>{summary.description}</Markdown>
          ) : (
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
        </>
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
      <span className="text-fg-muted text-xs">Started</span>
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

function getStatusLabel(status: AutopilotStatus): string {
  switch (status.status) {
    case "idle":
      return "Ready";
    case "server_side_processing":
      return "Thinking...";
    case "waiting_for_tool_call_authorization":
      return "Waiting";
    case "waiting_for_tool_execution":
      return "Executing tool...";
    case "waiting_for_retry":
      return "Something went wrong. Retrying...";
    case "failed":
      return "Something went wrong. Please try again.";
  }
}

function isLoadingStatus(status: AutopilotStatus): boolean {
  return (
    status.status === "server_side_processing" ||
    status.status === "waiting_for_tool_execution" ||
    status.status === "waiting_for_retry"
  );
}

function StatusIndicator({ status }: { status: AutopilotStatus }) {
  const showSpinner = isLoadingStatus(status);
  return (
    <div className="flex items-center gap-4 py-2">
      <div className="border-border flex-1 border-t" />
      <span className="text-fg-muted flex items-center gap-1.5 text-xs">
        {getStatusLabel(status)}
        {showSpinner && <Loader2 className="h-3 w-3 animate-spin" />}
      </span>
      <div className="border-border flex-1 border-t" />
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
  status,
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

      {/* Status indicator at the bottom */}
      {status && <StatusIndicator status={status} />}
    </div>
  );
}
