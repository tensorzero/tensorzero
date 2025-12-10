import { data, type LoaderFunctionArgs } from "react-router";
import { queryModelInferencesByInferenceId } from "~/utils/clickhouse/inference.server";
import {
  pollForFeedbackItem,
  queryLatestFeedbackIdByMetric,
} from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import { getUsedVariants } from "~/utils/clickhouse/function";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { logger } from "~/utils/logger";
import type { InferenceDetailData } from "~/components/inference/InferenceDetailContent";
import { getTensorZeroClient } from "~/utils/get-tensorzero-client.server";
import { loadFileDataForStoredInput } from "~/utils/resolve.server";

export async function loader({
  request,
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { inference_id } = params;
  const url = new URL(request.url);
  const newFeedbackId = url.searchParams.get("newFeedbackId");

  if (!inference_id) {
    throw data("Inference ID is required", { status: 400 });
  }

  try {
    const dbClient = await getNativeDatabaseClient();
    const tensorZeroClient = getTensorZeroClient();

    const inferencesPromise = tensorZeroClient.getInferences({
      ids: [inference_id],
      output_source: "inference",
    });
    const modelInferencesPromise =
      queryModelInferencesByInferenceId(inference_id);
    const demonstrationFeedbackPromise =
      dbClient.queryDemonstrationFeedbackByInferenceId({
        inference_id,
        limit: 1,
      });

    // If there is a freshly inserted feedback, ClickHouse may take some time to
    // update the feedback table and materialized views as it is eventually consistent.
    // In this case, we poll for the feedback item until it is found.
    const feedbackDataPromise = newFeedbackId
      ? pollForFeedbackItem(inference_id, newFeedbackId, 10)
      : dbClient.queryFeedbackByTargetId({
          target_id: inference_id,
          limit: 10,
        });

    let inferences,
      model_inferences,
      demonstration_feedback,
      feedback_bounds,
      feedback,
      latestFeedbackByMetric;

    if (newFeedbackId) {
      // When there's new feedback, wait for polling to complete before querying
      // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
      [inferences, model_inferences, demonstration_feedback, feedback] =
        await Promise.all([
          inferencesPromise,
          modelInferencesPromise,
          demonstrationFeedbackPromise,
          feedbackDataPromise,
        ]);

      // Query these after polling completes to avoid race condition with materialized views
      [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
        dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
        queryLatestFeedbackIdByMetric({ target_id: inference_id }),
      ]);
    } else {
      // Normal case: execute all queries in parallel
      [
        inferences,
        model_inferences,
        demonstration_feedback,
        feedback_bounds,
        feedback,
        latestFeedbackByMetric,
      ] = await Promise.all([
        inferencesPromise,
        modelInferencesPromise,
        demonstrationFeedbackPromise,
        dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
        feedbackDataPromise,
        queryLatestFeedbackIdByMetric({ target_id: inference_id }),
      ]);
    }

    if (inferences.inferences.length !== 1) {
      throw data(`No inference found for id ${inference_id}.`, {
        status: 404,
      });
    }
    const inference = inferences.inferences[0];
    const resolvedInput = await loadFileDataForStoredInput(inference.input);

    // Get used variants for default function
    const usedVariants =
      inference.function_name === DEFAULT_FUNCTION
        ? await getUsedVariants(inference.function_name)
        : [];

    const inferenceData: InferenceDetailData = {
      inference,
      input: resolvedInput,
      model_inferences,
      feedback,
      feedback_bounds,
      hasDemonstration: demonstration_feedback.length > 0,
      latestFeedbackByMetric,
      usedVariants,
    };

    return Response.json(inferenceData);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch inference:", error);
    throw data("Failed to fetch inference details", { status: 500 });
  }
}
