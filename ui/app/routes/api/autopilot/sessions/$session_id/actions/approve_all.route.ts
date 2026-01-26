import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

type ApproveAllRequest = {
  last_tool_call_event_id: string;
};

/**
 * API route for approving all pending tool calls in a session.
 *
 * Route: POST /api/autopilot/sessions/:session_id/actions/approve_all
 *
 * Request body:
 * - last_tool_call_event_id: string - Only approve tool calls with event IDs <= this value
 *
 * Response:
 * - approved_count: number - Number of tool calls that were approved
 * - event_ids: string[] - Event IDs of the newly created authorization events
 * - tool_call_event_ids: string[] - Event IDs of the tool calls that were approved
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: ApproveAllRequest;
  try {
    body = (await request.json()) as ApproveAllRequest;
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  if (!body.last_tool_call_event_id) {
    return new Response("last_tool_call_event_id is required", { status: 400 });
  }

  const client = getAutopilotClient();

  try {
    const response = await client.approveAllToolCalls(sessionId, {
      last_tool_call_event_id: body.last_tool_call_event_id,
    });

    return Response.json(response);
  } catch (error) {
    logger.error("Failed to approve all tool calls:", error);
    return new Response("Failed to approve all tool calls", { status: 500 });
  }
}
