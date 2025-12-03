import {
  data,
  type LoaderFunctionArgs,
  type ActionFunctionArgs,
} from "react-router";
import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference.server";
import { queryLatestFeedbackIdByMetric } from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import { getUsedVariants } from "~/utils/clickhouse/function";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";
import type { InferenceDetailData } from "~/components/inference/InferenceDetailContent";

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { inference_id } = params;

  if (!inference_id) {
    throw data("Inference ID is required", { status: 400 });
  }

  try {
    const dbClient = await getNativeDatabaseClient();

    // Fetch all data in parallel
    const [
      inference,
      model_inferences,
      feedback,
      feedback_bounds,
      demonstration_feedback,
      latestFeedbackByMetric,
    ] = await Promise.all([
      queryInferenceById(inference_id),
      queryModelInferencesByInferenceId(inference_id),
      dbClient.queryFeedbackByTargetId({
        target_id: inference_id,
        limit: 10,
      }),
      dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
      dbClient.queryDemonstrationFeedbackByInferenceId({
        inference_id: inference_id,
        limit: 1,
      }),
      queryLatestFeedbackIdByMetric({ target_id: inference_id }),
    ]);

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
