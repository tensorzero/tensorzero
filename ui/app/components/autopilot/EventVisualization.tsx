import { AlertTriangle } from "lucide-react";
import { ReadOnlyCodeBlock } from "~/components/ui/markdown";
import type {
  GatewayEvent,
  TopKEvaluationVisualization,
  VisualizationType,
} from "~/types/tensorzero";
import TopKEvaluationViz from "./TopKEvaluationViz";
import FeedbackByVariantChart, {
  type ParsedFeedbackByVariant,
  parseFeedbackByVariant,
  parseFeedbackChartData,
} from "./FeedbackByVariantChart";

export type EventVisualizationData =
  | { type: "top_k_evaluation"; data: TopKEvaluationVisualization }
  | { type: "feedback_by_variant"; data: ParsedFeedbackByVariant[] }
  | { type: "unknown"; raw: unknown };

function isTopKEvaluation(
  visualization: VisualizationType,
): visualization is { type: "top_k_evaluation" } & TopKEvaluationVisualization {
  return (
    typeof visualization === "object" &&
    visualization !== null &&
    "type" in visualization &&
    visualization.type === "top_k_evaluation"
  );
}

export function detectEventVisualization(
  event: GatewayEvent,
): EventVisualizationData | null {
  if (event.payload.type === "visualization") {
    const visualization = event.payload.visualization;
    if (isTopKEvaluation(visualization)) {
      return { type: "top_k_evaluation", data: visualization };
    }
    return { type: "unknown", raw: visualization };
  }

  if (
    event.payload.type === "tool_result" &&
    event.payload.outcome.type === "success"
  ) {
    const outcome = event.payload.outcome;
    const feedbackData =
      "result_value" in outcome
        ? parseFeedbackByVariant(outcome.result_value)
        : parseFeedbackChartData(outcome.result);
    if (feedbackData) {
      return { type: "feedback_by_variant", data: feedbackData };
    }
  }

  return null;
}

function UnknownVisualizationFallback({ raw }: { raw: unknown }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-fg-muted flex items-center gap-2 text-sm">
        <AlertTriangle className="h-4 w-4 text-yellow-600" />
        <span>
          Unknown visualization type. Your TensorZero deployment may be
          outdated.
        </span>
      </div>
      <ReadOnlyCodeBlock code={JSON.stringify(raw, null, 2)} language="json" />
    </div>
  );
}

export default function EventVisualization({
  data,
}: {
  data: EventVisualizationData;
}) {
  switch (data.type) {
    case "top_k_evaluation":
      return <TopKEvaluationViz data={data.data} />;
    case "feedback_by_variant":
      return <FeedbackByVariantChart data={data.data} />;
    case "unknown":
      return <UnknownVisualizationFallback raw={data.raw} />;
    default: {
      const _exhaustiveCheck: never = data;
      return _exhaustiveCheck;
    }
  }
}
