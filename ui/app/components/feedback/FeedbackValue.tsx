import { Code } from "~/components/ui/code";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import type { MetricConfig } from "~/utils/config/metric";

export default function FeedbackValue({
  feedback,
  metric,
}: {
  feedback: FeedbackRow;
  metric?: MetricConfig;
}) {
  // Handle boolean metrics
  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : "unknown";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    const failure =
      (feedback.value === true && optimize === "min") ||
      (feedback.value === false && optimize === "max");

    return (
      <div className="flex items-center gap-2">
        <div
          className={`h-2 w-2 rounded-full ${
            success ? "bg-green-700" : failure ? "bg-red-700" : "bg-gray-700"
          }`}
        />
        <span>{feedback.value ? "True" : "False"}</span>
      </div>
    );
  }

  // Handle float metrics
  if (feedback.type === "float" && typeof feedback.value === "number") {
    return <div>{feedback.value.toFixed(3)}</div>;
  }

  // Handle comments and demonstrations (both have string values)
  if (feedback.type === "comment" && typeof feedback.value === "string") {
    return (
      <div className="whitespace-pre-wrap break-words">{feedback.value}</div>
    );
  }

  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    // truncate to 1000 characters
    return (
      <Code className="text-sm">
        {feedback.value.length > 1000
          ? feedback.value.slice(0, 1000) + "..."
          : feedback.value}
      </Code>
    );
  }

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
