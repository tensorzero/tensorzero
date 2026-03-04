import type { GatewayEvent } from "~/types/tensorzero";
import FeedbackByVariantChart, {
  type ParsedFeedbackByVariant,
  parseFeedbackChartData,
} from "./FeedbackByVariantChart";

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
