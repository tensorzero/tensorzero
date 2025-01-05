import type { FeedbackRow } from "~/utils/clickhouse/feedback";

export default function FeedbackValue({ feedback }: { feedback: FeedbackRow }) {
  // Handle boolean metrics
  if ("metric_name" in feedback && typeof feedback.value === "boolean") {
    return <div>{feedback.value ? "✅ True" : "❌ False"}</div>;
  }

  // Handle float metrics
  if ("metric_name" in feedback && typeof feedback.value === "number") {
    return <div>{feedback.value.toFixed(3)}</div>;
  }

  // Handle comments and demonstrations (both have string values)
  if ("value" in feedback && typeof feedback.value === "string") {
    return (
      <div className="whitespace-pre-wrap break-words">{feedback.value}</div>
    );
  }

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
