import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

/**
 * API route for cancelling an autopilot session.
 *
 * Route: POST /api/autopilot/sessions/:session_id/actions/cancel
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const client = getAutopilotClient();

  try {
    await client.cancelAutopilotSession(sessionId);
    return Response.json({ success: true });
  } catch (error) {
    logger.error("Failed to cancel session:", error);
    const message =
      error instanceof Error ? error.message : "Failed to cancel session";
    return Response.json({ success: false, error: message }, { status: 500 });
  }
}
