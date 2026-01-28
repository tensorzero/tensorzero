import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

/**
 * API route for interrupting an autopilot session.
 *
 * Route: POST /api/autopilot/sessions/:session_id/actions/interrupt
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return Response.json(
      { success: false, error: "Session ID is required" },
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      { success: false, error: "Method not allowed" },
      { status: 405 },
    );
  }

  const client = getAutopilotClient();

  try {
    await client.interruptAutopilotSession(sessionId);
    return Response.json({ success: true });
  } catch (error) {
    logger.error("Failed to interrupt session:", error);
    const message =
      error instanceof Error ? error.message : "Failed to interrupt session";
    return Response.json({ success: false, error: message }, { status: 500 });
  }
}
