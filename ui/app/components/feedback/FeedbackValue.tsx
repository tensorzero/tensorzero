import { Code } from "~/components/ui/code";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import { X, Check } from "lucide-react";
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
    const optimize = metric?.type === "boolean" ? metric.optimize : "max";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    return (
      <div className="flex items-center">
        <span className="flex items-center">
          {feedback.value ? "True" : "False"}
          {success ? (
            <Check className="ml-1 h-3 w-3 flex-shrink-0 text-green-700" />
          ) : (
            <X className="ml-1 h-3 w-3 flex-shrink-0 text-red-700" />
          )}
        </span>
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
      <Code>
        {feedback.value.length > 1000
          ? feedback.value.slice(0, 1000) + "..."
          : feedback.value}
      </Code>
    );
  }

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
