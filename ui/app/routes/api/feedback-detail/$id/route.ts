import { data, type LoaderFunctionArgs } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import type { ResolvedObject } from "~/types/tensorzero";

const FEEDBACK_TYPES = new Set([
  "boolean_feedback",
  "float_feedback",
  "comment_feedback",
  "demonstration_feedback",
]);

function isFeedbackType(
  type: ResolvedObject["type"],
): type is
  | "boolean_feedback"
  | "float_feedback"
  | "comment_feedback"
  | "demonstration_feedback" {
  return FEEDBACK_TYPES.has(type);
}

export type FeedbackType = "boolean" | "float" | "comment" | "demonstration";

function resolvedTypeToFeedbackType(
  type:
    | "boolean_feedback"
    | "float_feedback"
    | "comment_feedback"
    | "demonstration_feedback",
): FeedbackType {
  switch (type) {
    case "boolean_feedback":
      return "boolean";
    case "float_feedback":
      return "float";
    case "comment_feedback":
      return "comment";
    case "demonstration_feedback":
      return "demonstration";
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}

export interface FeedbackDetailData {
  id: string;
  feedback_type: FeedbackType;
}

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { id } = params;

  if (!id) {
    throw data("Feedback ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();
    const resolved = await client.resolveUuid(id);

    const feedbackObj = resolved.object_types.find((obj) =>
      isFeedbackType(obj.type),
    );

    if (!feedbackObj || !isFeedbackType(feedbackObj.type)) {
      throw data(`UUID ${id} is not a feedback`, { status: 404 });
    }

    const feedbackDetailData: FeedbackDetailData = {
      id,
      feedback_type: resolvedTypeToFeedbackType(feedbackObj.type),
    };

    return Response.json(feedbackDetailData);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch feedback detail:", error);
    throw data("Failed to fetch feedback details", { status: 500 });
  }
}
