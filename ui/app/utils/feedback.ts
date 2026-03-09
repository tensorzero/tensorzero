import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedbackBounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
};

export interface GroupedFeedback {
  metrics: FeedbackRow[];
  comments: Extract<FeedbackRow, { type: "comment" }>[];
  demonstrations: Extract<FeedbackRow, { type: "demonstration" }>[];
}

export function groupFeedbackByType(feedback: FeedbackRow[]): GroupedFeedback {
  const metrics: FeedbackRow[] = [];
  const comments: Extract<FeedbackRow, { type: "comment" }>[] = [];
  const demonstrations: Extract<FeedbackRow, { type: "demonstration" }>[] = [];
  for (const f of feedback) {
    if (f.type === "boolean" || f.type === "float") {
      metrics.push(f);
    } else if (f.type === "comment") {
      comments.push(f);
    } else if (f.type === "demonstration") {
      demonstrations.push(f);
    }
  }
  return { metrics, comments, demonstrations };
}

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
