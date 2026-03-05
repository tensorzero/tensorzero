import type { MetricConfig, FeedbackRow } from "~/types/tensorzero";
import { BooleanItem, FloatItem } from "./FeedbackValueItem";

interface FeedbackValueProps {
  feedback: FeedbackRow;
  metric?: MetricConfig;
}

export default function FeedbackValue({
  feedback,
  metric,
}: FeedbackValueProps) {
  const isHumanFeedback =
    feedback.tags["tensorzero::human_feedback"] === "true";

  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : "unknown";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    const failure =
      (feedback.value === true && optimize === "min") ||
      (feedback.value === false && optimize === "max");

    let status: "success" | "failure" | "default" = "default";

    if (success) {
      status = "success";
    } else if (failure) {
      status = "failure";
    }

    return (
      <BooleanItem
        value={feedback.value}
        status={status}
        isHumanFeedback={isHumanFeedback}
      />
    );
  }

  if (feedback.type === "float" && typeof feedback.value === "number") {
    return (
      <FloatItem value={feedback.value} isHumanFeedback={isHumanFeedback} />
    );
  }

  return <div className="text-red-500">Invalid feedback type</div>;
}
