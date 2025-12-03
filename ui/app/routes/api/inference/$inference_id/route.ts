import {
  data,
  type LoaderFunctionArgs,
  type ActionFunctionArgs,
} from "react-router";
import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference.server";
import {
  pollForFeedbackItem,
  queryLatestFeedbackIdByMetric,
} from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import { getUsedVariants } from "~/utils/clickhouse/function";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";
import type { InferenceDetailData } from "~/components/inference/InferenceDetailContent";

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

    const inferencePromise = queryInferenceById(inference_id);
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

    let inference,
      model_inferences,
      demonstration_feedback,
      feedback_bounds,
      feedback,
      latestFeedbackByMetric;

    if (newFeedbackId) {
      // When there's new feedback, wait for polling to complete before querying
      // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
      [inference, model_inferences, demonstration_feedback, feedback] =
        await Promise.all([
          inferencePromise,
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
        inference,
        model_inferences,
        demonstration_feedback,
        feedback_bounds,
        feedback,
        latestFeedbackByMetric,
      ] = await Promise.all([
        inferencePromise,
        modelInferencesPromise,
        demonstrationFeedbackPromise,
        dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
        feedbackDataPromise,
        queryLatestFeedbackIdByMetric({ target_id: inference_id }),
      ]);
    }

    if (!inference) {
      throw data(`Inference ${inference_id} not found`, { status: 404 });
    }

    // Get used variants for default function
    const usedVariants =
      inference.function_name === DEFAULT_FUNCTION
        ? await getUsedVariants(inference.function_name)
        : [];

    const inferenceData: InferenceDetailData = {
      inference,
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

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

export async function action({ request, params }: ActionFunctionArgs) {
  const { inference_id } = params;
  const formData = await request.formData();
  const _action = formData.get("_action");

  switch (_action) {
    case "addFeedback": {
      try {
        const response = await addHumanFeedback(formData);
        // Return success with the new feedback ID
        return data<ActionData>({
          redirectTo: `/api/inference/${inference_id}?newFeedbackId=${response.feedback_id}`,
        });
      } catch (error) {
        if (isTensorZeroServerError(error)) {
          return data<ActionData>(
            { error: error.message },
            { status: error.status },
          );
        }
        return data<ActionData>(
          { error: "Unknown server error. Try again." },
          { status: 500 },
        );
      }
    }

    case null:
      logger.error("No action provided");
      return data<ActionData>({ error: "No action provided" }, { status: 400 });

    default:
      logger.error(`Unknown action: ${_action}`);
      return data<ActionData>(
        { error: `Unknown action: ${_action}` },
        { status: 400 },
      );
  }
}
