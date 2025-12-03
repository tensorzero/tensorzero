import { data, type ActionFunctionArgs } from "react-router";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

/**
 * Dedicated API route for adding human feedback to inferences.
 *
 * This centralizes all inference feedback handling in one place, used by:
 * - InferencePreviewSheet (side panel on episode page)
 * - Inference detail page
 * - Any future components that need to add feedback
 *
 * The route returns a redirectTo URL that includes the newFeedbackId,
 * which allows the caller to poll for the feedback (ClickHouse eventual consistency).
 */
export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  const inferenceId = formData.get("inferenceId");
  if (!inferenceId || typeof inferenceId !== "string") {
    return data<ActionData>(
      { error: "inferenceId is required" },
      { status: 400 },
    );
  }

  try {
    const response = await addHumanFeedback(formData);
    // Return the API URL with newFeedbackId for polling
    // The caller (InferenceDetailContent) uses this URL to refresh data
    return data<ActionData>({
      redirectTo: `/api/inference/${inferenceId}?newFeedbackId=${response.feedback_id}`,
    });
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
