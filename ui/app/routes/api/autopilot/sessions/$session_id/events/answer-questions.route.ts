import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import type { UserQuestionAnswer } from "~/types/tensorzero";
import { logger } from "~/utils/logger";

type AnswerQuestionsRequest = {
  user_questions_event_id: string;
  responses: Record<string, UserQuestionAnswer>;
};

/**
 * API route for submitting user question responses.
 *
 * Route: POST /api/autopilot/sessions/:session_id/events/answer-questions
 *
 * Request body:
 * - user_questions_event_id: string - ID of the user_questions event to respond to
 * - responses: Record<string, UserQuestionAnswer> - Map from question ID to response
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: AnswerQuestionsRequest;
  try {
    body = (await request.json()) as AnswerQuestionsRequest;
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  if (!body.user_questions_event_id) {
    return new Response("user_questions_event_id is required", { status: 400 });
  }

  if (!body.responses || typeof body.responses !== "object") {
    return new Response("responses is required", { status: 400 });
  }

  const client = getAutopilotClient();

  try {
    const response = await client.createAutopilotEvent(sessionId, {
      payload: {
        type: "user_questions_answers",
        user_questions_event_id: body.user_questions_event_id,
        responses: body.responses,
      },
    });

    return Response.json(response);
  } catch (error) {
    logger.error("Failed to create question response event:", error);
    return new Response("Failed to submit question responses", {
      status: 500,
    });
  }
}
