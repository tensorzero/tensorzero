import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  resolveModelInferences,
  loadFileDataForStoredInput,
} from "~/utils/resolve.server";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import type {
  FeedbackRow,
  FeedbackBounds,
  StoredInference,
  Input,
} from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";

// Types for streamed data
export type ModelInferencesData = ParsedModelInferenceRow[];

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedback_bounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
};

// Fetch functions for independent streaming
export async function fetchModelInferences(
  inference_id: string,
): Promise<ModelInferencesData> {
  const tensorZeroClient = getTensorZeroClient();
  const response = await tensorZeroClient.getModelInferences(inference_id);
  return resolveModelInferences(response.model_inferences);
}

export async function fetchUsedVariants(
  functionName: string,
): Promise<string[]> {
  if (functionName !== DEFAULT_FUNCTION) {
    return [];
  }
  const tensorZeroClient = getTensorZeroClient();
  return tensorZeroClient.getUsedVariants(functionName);
}

export async function fetchHasDemonstration(
  inference_id: string,
): Promise<boolean> {
  const tensorZeroClient = getTensorZeroClient();
  const demonstrationFeedback = await tensorZeroClient.getDemonstrationFeedback(
    inference_id,
    { limit: 1 },
  );
  return demonstrationFeedback.length > 0;
}

export async function fetchInput(
  inference: StoredInference,
): Promise<Input | undefined> {
  if (!inference.input) {
    return undefined;
  }
  return loadFileDataForStoredInput(inference.input);
}

export async function fetchFeedbackData(
  inference_id: string,
  params: {
    newFeedbackId: string | null;
    beforeFeedback: string | null;
    afterFeedback: string | null;
    limit: number;
  },
): Promise<FeedbackData> {
  const tensorZeroClient = getTensorZeroClient();
  const { newFeedbackId, beforeFeedback, afterFeedback, limit } = params;

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but eventually time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  if (newFeedbackId) {
    // Sequential case: poll first, then query bounds/metrics
    const feedback = await pollForFeedbackItem(
      inference_id,
      newFeedbackId,
      limit,
    );
    const [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ]);
    return { feedback, feedback_bounds, latestFeedbackByMetric };
  }

  // Normal case: execute all queries in parallel
  const [feedback, feedback_bounds, latestFeedbackByMetric] = await Promise.all(
    [
      tensorZeroClient.getFeedbackByTargetId(inference_id, {
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      }),
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ],
  );
  return { feedback, feedback_bounds, latestFeedbackByMetric };
}
