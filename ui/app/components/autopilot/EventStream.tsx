import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  ChevronRight,
  RotateCcw,
} from "lucide-react";
import { Button } from "~/components/ui/button";
import { Component, type RefObject, useState } from "react";
import {
  AnimatedEllipsis,
  EllipsisMode,
} from "~/components/ui/AnimatedEllipsis";
import { Markdown, ReadOnlyCodeBlock } from "~/components/ui/markdown";
import { Skeleton } from "~/components/ui/skeleton";
import { logger } from "~/utils/logger";
import { DotSeparator } from "~/components/ui/DotSeparator";
import { TableItemTime } from "~/components/ui/TableItems";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { useAutopilotSession } from "~/contexts/AutopilotSessionContext";
import type {
  AutopilotStatus,
  EventPayloadMessageContent,
  GatewayEvent,
  GatewayEventPayload,
  TopKEvaluationVisualization,
  VisualizationType,
} from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { ApplyConfigChangeButton } from "~/components/autopilot/ApplyConfigChangeButton";
import TopKEvaluationViz from "./TopKEvaluationViz";

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
  loadError?: string | null;
  onRetryLoad?: () => void;
  topSentinelRef?: RefObject<HTMLDivElement | null>;
  pendingToolCallIds?: Set<string>;
  optimisticMessages?: OptimisticMessage[];
  status?: AutopilotStatus;
  configWriteEnabled?: boolean;
  sessionId?: string;
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
  {
    type:
      | "tool_call"
      | "tool_call_authorization"
      | "tool_result"
      | "visualization";
  }
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
    event.payload.type === "tool_result" ||
    event.payload.type === "visualization"
  );
}

/**
 * Extracts the tool execution ID from a tool event.
 * For tool_call events, this is in side_info.tool_call_event_id.
 * For tool_call_authorization and tool_result events, this is tool_call_event_id on the payload.
 * For visualization events, this is tool_execution_id on the payload.
 */
export function getToolCallEventId(event: ToolEvent): string {
  const { payload } = event;
  if (payload.type === "tool_call") {
    return payload.side_info.tool_call_event_id;
  }
  if (payload.type === "visualization") {
    return payload.tool_execution_id;
  }
  return payload.tool_call_event_id;
}

/**
 * Type guard to check if an event is a config write event.
 * A config write event is a tool_call with name === "write_config".
 */
export function isConfigWriteEvent(event: GatewayEvent): boolean {
  return (
    event.payload.type === "tool_call" && event.payload.name === "write_config"
  );
}

function getMessageText(content: EventPayloadMessageContent[]) {
  return content.map((cb) => cb.text).join("\n\n");
}

/**
 * Get the title for a visualization based on its type.
 */
function getVisualizationTitle(visualization: VisualizationType): string {
  if (typeof visualization !== "object" || visualization === null) {
    return "Visualization";
  }
  if ("type" in visualization) {
    if (visualization.type === "top_k_evaluation") {
      return "Top-K Evaluation Results";
    }
    // Unknown visualization type with a type field
    return `Visualization (${String(visualization.type)})`;
  }
  return "Visualization";
}

/**
 * Renders the appropriate visualization component based on the type.
 */
function VisualizationRenderer({
  visualization,
}: {
  visualization: VisualizationType;
}) {
  // Check for known visualization types
  if (
    typeof visualization === "object" &&
    visualization !== null &&
    "type" in visualization &&
    visualization.type === "top_k_evaluation"
  ) {
    // Type assertion needed because TypeScript can't narrow through the untagged union
    return (
      <TopKEvaluationViz data={visualization as TopKEvaluationVisualization} />
    );
  }

  // Unknown or malformed visualization - show raw JSON with a warning
  return (
    <div className="flex flex-col gap-2">
      <div className="text-fg-muted flex items-center gap-2 text-sm">
        <AlertTriangle className="h-4 w-4 text-yellow-600" />
        <span>
          Unknown visualization type. Your TensorZero deployment may be
          outdated.
        </span>
      </div>
      <ReadOnlyCodeBlock
        code={JSON.stringify(visualization, null, 2)}
        language="json"
      />
    </div>
  );
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
    case "tool_call": {
      return { description: JSON.stringify(payload.arguments, null, 2) };
    }
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
      return {
        description: payload.message,
      };
    case "visualization":
      // Visualization events render their own content, no text description needed
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
      return roleLabel;
    }
    case "status_update":
      return "Status Update";
    case "tool_call":
      return (
        <span className="inline-flex items-center gap-2">
          Tool Call
          <DotSeparator />
          <span className="font-mono font-medium">{payload.name}</span>
        </span>
      );
    case "tool_call_authorization":
      switch (payload.status.type) {
        case "approved":
          return (
            <span className="inline-flex items-center gap-2">
              Tool Call Authorization
              <DotSeparator />
              Approved
            </span>
          );
        case "rejected":
          return (
            <span className="inline-flex items-center gap-2">
              Tool Call Authorization
              <DotSeparator />
              Rejected
            </span>
          );
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
          return (
            <span className="inline-flex items-center gap-2">
              Tool Result
              <DotSeparator />
              Success
            </span>
          );
        case "failure":
          // TODO: need tool name
          return (
            <span className="inline-flex items-center gap-2">
              Tool Result
              <DotSeparator />
              Failure
            </span>
          );
        case "rejected":
          // TODO: need tool name
          return (
            <span className="inline-flex items-center gap-2">
              Tool Result
              <DotSeparator />
              Rejected
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
              Tool Result
              <DotSeparator />
              Missing Tool
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
              Tool Result
              <DotSeparator />
              Unknown
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
    case "visualization":
      return (
        <span className="inline-flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          <span>{getVisualizationTitle(payload.visualization)}</span>
        </span>
      );
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

/**
 * Error boundary for individual event items.
 * Prevents a single malformed event from crashing the entire chat.
 */
interface EventErrorBoundaryState {
  hasError: boolean;
}

interface EventErrorBoundaryProps {
  eventId: string;
  children: React.ReactNode;
}

class EventErrorBoundary extends Component<
  EventErrorBoundaryProps,
  EventErrorBoundaryState
> {
  constructor(props: EventErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): EventErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    logger.error(
      `Event ${this.props.eventId} failed to render:`,
      error,
      errorInfo,
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="border-border bg-bg-secondary rounded-md border px-4 py-3">
          <div className="flex items-center gap-2 text-sm">
            <AlertCircle className="h-4 w-4 text-amber-500" />
            <span className="text-fg-muted">
              Failed to display event. The event data may be corrupted.
            </span>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function EventItem({
  event,
  isPending = false,
  configWriteEnabled = false,
  sessionId,
}: {
  event: GatewayEvent;
  isPending?: boolean;
  configWriteEnabled?: boolean;
  sessionId?: string;
}) {
  const { yoloMode } = useAutopilotSession();
  const summary = summarizeEvent(event);
  const title = renderEventTitle(event);
  const eventIsToolEvent = isToolEvent(event);
  const isConfigWrite = isConfigWriteEvent(event);
  const isExpandable =
    event.payload.type === "tool_call" ||
    event.payload.type === "error" ||
    event.payload.type === "visualization" ||
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
            {isPending && !yoloMode && (
              <span className="rounded bg-blue-200 px-1.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-800 dark:text-blue-200">
                Action Required
              </span>
            )}
          </button>
        ) : (
          label
        )}
        <div className="text-fg-muted flex items-center gap-1.5 text-xs">
          {isConfigWrite && configWriteEnabled && sessionId && (
            <ApplyConfigChangeButton sessionId={sessionId} event={event} />
          )}
          {eventIsToolEvent && (
            <>
              <ToolEventId id={getToolCallEventId(event)} />
              <DotSeparator />
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
          ) : event.payload.type === "tool_call" ? (
            <ReadOnlyCodeBlock code={summary.description} language="json" />
          ) : (
            <p
              className={cn(
                "text-fg-secondary text-sm whitespace-pre-wrap",
                (event.payload.type === "tool_result" ||
                  event.payload.type === "error") &&
                  "font-mono",
              )}
            >
              {summary.description}
            </p>
          )}
        </>
      )}
      {shouldShowDetails && event.payload.type === "visualization" && (
        <VisualizationRenderer visualization={event.payload.visualization} />
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

function Divider({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-5 py-2">
      <div className="border-border flex-1 border-t" />
      <span className="text-fg-muted relative text-xs">{children}</span>
      <div className="border-border flex-1 border-t" />
    </div>
  );
}

function SessionStartDivider() {
  return <Divider>Start</Divider>;
}

function OptimisticMessageItem({ message }: { message: OptimisticMessage }) {
  // Optimistic messages are shown after POST succeeds, so the message is saved.
  // We display it like a regular user message, with a skeleton for the timestamp.
  return (
    <div className="border-border bg-bg-secondary flex flex-col gap-2 rounded-md border px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        <span className="text-sm font-medium">User</span>
        <Skeleton className="h-4 w-32" />
      </div>
      <p className="text-fg-secondary text-sm whitespace-pre-wrap">
        {message.text}
      </p>
    </div>
  );
}

function getStatusLabel(status: AutopilotStatus): {
  text: string;
  showEllipsis: boolean;
} {
  switch (status.status) {
    case "idle":
      return { text: "Ready", showEllipsis: false };
    case "server_side_processing":
      return { text: "Thinking", showEllipsis: true };
    case "waiting_for_tool_call_authorization":
      return { text: "Waiting", showEllipsis: false };
    case "waiting_for_tool_execution":
      return { text: "Executing tool", showEllipsis: true };
    case "waiting_for_retry":
      return { text: "Something went wrong. Retrying", showEllipsis: true };
    case "failed":
      return {
        text: "Something went wrong. Please try again.",
        showEllipsis: false,
      };
  }
}

function StatusIndicator({ status }: { status: AutopilotStatus }) {
  const { text, showEllipsis } = getStatusLabel(status);
  return (
    <Divider>
      {text}
      {showEllipsis && <AnimatedEllipsis mode={EllipsisMode.Absolute} />}
    </Divider>
  );
}

function LoadErrorNotice({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex items-center justify-center gap-2 py-2 text-sm text-amber-600">
      <span>Failed to load older messages</span>
      {onRetry && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onRetry}
          className="h-6 gap-1 px-2 text-amber-600 hover:text-amber-700"
        >
          <RotateCcw className="h-3 w-3" />
          Retry
        </Button>
      )}
    </div>
  );
}

export default function EventStream({
  events,
  className,
  isLoadingOlder = false,
  hasReachedStart = false,
  loadError,
  onRetryLoad,
  topSentinelRef,
  pendingToolCallIds,
  optimisticMessages = [],
  status,
  configWriteEnabled = false,
  sessionId,
}: EventStreamProps) {
  // Determine what to show at the top: sentinel, error, or session start
  // Only show session start when there's content to display (events or optimistic messages)
  const showSessionStart =
    (hasReachedStart || optimisticMessages.length > 0) &&
    !isLoadingOlder &&
    !loadError &&
    (events.length > 0 || optimisticMessages.length > 0);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* Sentinel for loading more - always present unless showing session start */}
      {/* Must stay in DOM during loading/error so IntersectionObserver keeps working */}
      {!showSessionStart && (
        <div ref={topSentinelRef} className="h-1" aria-hidden="true" />
      )}

      {/* Error state - show retry notice (after sentinel so it appears below) */}
      {loadError && <LoadErrorNotice onRetry={onRetryLoad} />}

      {/* Session start indicator */}
      {showSessionStart && <SessionStartDivider />}

      {/* Show skeletons when more content exists above (not yet loaded) */}
      {/* This prevents layout jump when loading starts */}
      {!showSessionStart && !hasReachedStart && !loadError && (
        <EventSkeletons count={3} />
      )}

      {events.map((event) => (
        <EventErrorBoundary key={event.id} eventId={event.id}>
          <EventItem
            event={event}
            isPending={pendingToolCallIds?.has(event.id)}
            configWriteEnabled={configWriteEnabled}
            sessionId={sessionId}
          />
        </EventErrorBoundary>
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
