import type { GatewayEvent } from "~/types/tensorzero";
import FeedbackByVariantChart, {
  type ParsedFeedbackByVariant,
  parseFeedbackChartData,
} from "./FeedbackByVariantChart";

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
): ParsedFeedbackByVariant[] | null {
  if (event.payload.type !== "tool_result") return null;
  if (event.payload.outcome.type !== "success") return null;

  return parseFeedbackChartData(event.payload.outcome.result);
}

export function ToolResultVisualization({
  data,
}: {
  data: ParsedFeedbackByVariant[];
}) {
  return <FeedbackByVariantChart data={data} />;
}
