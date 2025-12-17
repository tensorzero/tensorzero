import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import { logger } from "~/utils/logger";
import type { FeedbackRow } from "~/types/tensorzero";

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
  const dbClient = await getNativeDatabaseClient();

  let feedback: FeedbackRow[] = [];
  let found = false;
  for (let i = 0; i < maxRetries; i++) {
    feedback = await dbClient.queryFeedbackByTargetId({
      target_id: targetId,
      before: undefined,
      after: undefined,
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
