import { Code } from "~/components/ui/code";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";

export default function FeedbackValue({ feedback }: { feedback: FeedbackRow }) {
  // Handle boolean metrics
  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    return <div>{feedback.value ? "✅ True" : "❌ False"}</div>;
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
