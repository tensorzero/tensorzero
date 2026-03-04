import type { GatewayEvent } from "~/types/tensorzero";
import type { FeedbackChartData } from "./FeedbackByVariantChart";
import FeedbackByVariantChart, {
  parseFeedbackChartData,
} from "./FeedbackByVariantChart";

/**
 * Resolves tool_call arguments for a given tool_call_event_id
 * by scanning the events array for the matching tool_call event.
 */
function resolveToolCallArguments(
  toolCallEventId: string,
  events: GatewayEvent[],
): unknown {
  const toolCall = events.find(
    (e) =>
      e.payload.type === "tool_call" &&
      e.payload.side_info.tool_call_event_id === toolCallEventId,
  );
  return toolCall?.payload.type === "tool_call"
    ? toolCall.payload.arguments
    : undefined;
}

/**
 * Detects whether a tool_result event contains data that can be rendered
 * as a visualization. Uses shape detection (parsing the result payload)
 * rather than checking the tool name.
 *
 * To add a new tool result visualization, add a detection case here
 * and a corresponding render branch in ToolResultVisualization.
 */
export function detectToolResultVisualization(
  event: GatewayEvent,
  events: GatewayEvent[],
): FeedbackChartData | null {
  if (event.payload.type !== "tool_result") return null;
  if (event.payload.outcome.type !== "success") return null;

  return parseFeedbackChartData(
    event.payload.outcome.result,
    resolveToolCallArguments(event.payload.tool_call_event_id, events),
  );
}

export function ToolResultVisualization({ data }: { data: FeedbackChartData }) {
  return <FeedbackByVariantChart {...data} />;
}
