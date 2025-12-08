import { data, type ActionFunctionArgs } from "react-router";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

/**
 * Dedicated API route for adding human feedback to inferences or episodes.
 *
 * This centralizes all feedback handling in one place, used by:
 * - InferencePreviewSheet (side panel on episode page)
 * - Inference detail page
 * - Episode detail page
 * - Any future components that need to add feedback
 *
 * Expected form data:
 * - metricName: string - The metric to provide feedback for
 * - value: string - The feedback value (format depends on metric type)
 * - inferenceId: string (optional) - For inference-level feedback
 * - episodeId: string (optional) - For episode-level feedback
 * Note: Exactly one of inferenceId or episodeId must be provided.
 *
 * The route returns a redirectTo URL that includes the newFeedbackId,
 * which allows the caller to poll for the feedback (ClickHouse eventual consistency).
 */
export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  const inferenceId = formData.get("inferenceId");
  const episodeId = formData.get("episodeId");

  // Validate that exactly one of inferenceId or episodeId is provided
  const hasInferenceId = inferenceId && typeof inferenceId === "string";
  const hasEpisodeId = episodeId && typeof episodeId === "string";

  if (hasInferenceId && hasEpisodeId) {
    return data<ActionData>(
      { error: "Only one of inferenceId or episodeId should be provided" },
      { status: 400 },
    );
  }

  if (!hasInferenceId && !hasEpisodeId) {
    return data<ActionData>(
      { error: "Either inferenceId or episodeId is required" },
      { status: 400 },
    );
  }

  try {
    const response = await addHumanFeedback(formData);

    // Return the appropriate URL with newFeedbackId for polling
    let redirectTo: string;
    if (hasInferenceId) {
      // For inference feedback, return the API URL for polling
      redirectTo = `/api/inference/${inferenceId}?newFeedbackId=${response.feedback_id}`;
    } else {
      // For episode feedback, return the episode page URL with newFeedbackId
      redirectTo = `/observability/episodes/${episodeId}?newFeedbackId=${response.feedback_id}`;
    }

    return data<ActionData>({ redirectTo });
  } catch (error) {
    if (isTensorZeroServerError(error)) {
      return data<ActionData>(
        { error: error.message },
        { status: error.status },
      );
    }
    logger.error("Failed to add feedback:", error);
    return data<ActionData>(
      { error: "Unknown server error. Try again." },
      { status: 500 },
    );
  }
}
