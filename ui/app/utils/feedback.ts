import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedbackBounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
};

export function filterToLatestFeedback(
  feedback: FeedbackRow[],
  feedbackBounds?: FeedbackBounds,
  latestFeedbackByMetric?: Record<string, string>,
): FeedbackRow[] {
  if (!feedbackBounds || !latestFeedbackByMetric) return feedback;
  return feedback.filter((item) => {
    if (item.type === "comment") {
      const lastId = feedbackBounds.by_type.comment.last_id;
      return lastId === undefined || item.id === lastId;
    }
    if (item.type === "demonstration") {
      const lastId = feedbackBounds.by_type.demonstration.last_id;
      return lastId === undefined || item.id === lastId;
    }
    return latestFeedbackByMetric[item.metric_name] === item.id;
  });
}
