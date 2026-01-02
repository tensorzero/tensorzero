import type { ActionFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import type { ToolCallAuthorizationStatus } from "~/types/tensorzero";
import { logger } from "~/utils/logger";

// Hardcoded deployment_id (temporary - will be removed soon)
const DEPLOYMENT_ID = "019b7bb4-bd08-76ec-875e-4d27d5eb3864";

// Get version from build-time constant or fallback
const TENSORZERO_VERSION =
  typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "unknown";

type AuthorizeRequest = {
  tool_call_event_id: string;
  status: ToolCallAuthorizationStatus;
};

/**
 * API route for submitting tool call authorization decisions.
 *
 * Route: POST /api/autopilot/sessions/:session_id/events/authorize
 *
 * Request body:
 * - tool_call_event_id: string - ID of the tool_call event to authorize
 * - status: { type: "approved" } | { type: "rejected", reason: string }
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: AuthorizeRequest;
  try {
    body = (await request.json()) as AuthorizeRequest;
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  if (!body.tool_call_event_id) {
    return new Response("tool_call_event_id is required", { status: 400 });
  }

  if (!body.status || !body.status.type) {
    return new Response("status is required", { status: 400 });
  }

  if (body.status.type !== "approved" && body.status.type !== "rejected") {
    return new Response("status.type must be 'approved' or 'rejected'", {
      status: 400,
    });
  }

  if (body.status.type === "rejected" && !body.status.reason) {
    return new Response("status.reason is required for rejected status", {
      status: 400,
    });
  }

  const client = getAutopilotClient();

  try {
    const response = await client.createAutopilotEvent(sessionId, {
      deployment_id: DEPLOYMENT_ID,
      tensorzero_version: TENSORZERO_VERSION,
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        tool_call_event_id: body.tool_call_event_id,
        status: body.status,
      },
    });

    return Response.json(response);
  } catch (error) {
    logger.error("Failed to create authorization event:", error);
    return new Response("Failed to create authorization event", {
      status: 500,
    });
  }
}
