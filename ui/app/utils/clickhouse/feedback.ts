import { logger } from "~/utils/logger";
import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

/**
 * Polls for a specific feedback item on the first page.
 * @param targetId The ID of the target (e.g., inference_id).
 * @param feedbackId The ID of the feedback item to find.
 * @param limit The number of items per page to fetch.
 * @param maxRetries Maximum number of polling attempts.
 * @param retryDelay Delay between retries in milliseconds.
 * @returns The fetched feedback list.
 */
export async function pollForFeedbackItem(
  targetId: string,
  feedbackId: string,
  limit: number,
  maxRetries: number = 10,
  retryDelay: number = 200,
): Promise<FeedbackRow[]> {
  const tensorZeroClient = getTensorZeroClient();

  let feedback: FeedbackRow[] = [];
  let found = false;
  for (let i = 0; i < maxRetries; i++) {
    feedback = await tensorZeroClient.getFeedbackByTargetId(targetId, {
      limit,
    });
    if (feedback.some((f) => f.id === feedbackId)) {
      found = true;
      break;
    }
    if (i < maxRetries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
    }
  }
  if (!found) {
    logger.warn(
      `Feedback ${feedbackId} for target ${targetId} not found after ${maxRetries} retries.`,
    );
  }
  return feedback;
}

/**
 * Filters a feedback list to only the latest entry per metric/comment/demonstration.
 * Uses feedback bounds for comment/demonstration and a per-metric latest ID map for metrics.
 */
export function filterToLatest(
  feedback: FeedbackRow[],
  bounds: FeedbackBounds,
  latestByMetric: Record<string, string>,
): FeedbackRow[] {
  return feedback.filter((item) => {
    if (item.type === "comment")
      return item.id === bounds.by_type.comment.last_id;
    if (item.type === "demonstration")
      return item.id === bounds.by_type.demonstration.last_id;
    return latestByMetric[item.metric_name] === item.id;
  });
}
