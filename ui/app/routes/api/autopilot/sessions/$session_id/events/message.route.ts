import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

// Hardcoded deployment_id (temporary - will be removed soon)
const DEPLOYMENT_ID = "019b7bb4-bd08-76ec-875e-4d27d5eb3864";

// Get version from build-time constant or fallback
const TENSORZERO_VERSION =
  typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "unknown";

type MessageRequest = {
  text: string;
  previous_user_message_event_id?: string;
};

/**
 * API route for sending user messages to an autopilot session.
 *
 * Route: POST /api/autopilot/sessions/:session_id/events/message
 *
 * Request body:
 * - text: string - The message text to send
 * - previous_user_message_event_id?: string - For idempotency (prevents duplicates on retry)
 *
 * Use session_id = "00000000-0000-0000-0000-000000000000" (nil UUID) to create a new session.
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: MessageRequest;
  try {
    body = (await request.json()) as MessageRequest;
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  if (!body.text || typeof body.text !== "string") {
    return new Response("text is required and must be a string", {
      status: 400,
    });
  }

  if (body.text.trim().length === 0) {
    return new Response("text cannot be empty", { status: 400 });
  }

  const client = getAutopilotClient();

  try {
    const response = await client.createAutopilotEvent(sessionId, {
      deployment_id: DEPLOYMENT_ID,
      tensorzero_version: TENSORZERO_VERSION,
      payload: {
        type: "message",
        role: "user",
        content: [
          {
            type: "text",
            text: body.text,
          },
        ],
      },
      previous_user_message_event_id: body.previous_user_message_event_id,
    });

    return Response.json(response);
  } catch (error) {
    logger.error("Failed to create user message event:", error);
    return new Response("Failed to send message", {
      status: 500,
    });
  }
}
