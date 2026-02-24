import type { ActionFunctionArgs, LoaderFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import type { CreateEventGatewayRequest } from "~/types/tensorzero";

/**
 * API route for creating an autopilot event.
 *
 * Route: POST /api/autopilot/sessions/:session_id/events
 *
 * Request body: a CreateEventGatewayRequest to forward to the autopilot API.
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: CreateEventGatewayRequest;
  try {
    body = (await request.json()) as CreateEventGatewayRequest;
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const client = getAutopilotClient();

  try {
    const response = await client.createAutopilotEvent(sessionId, body);
    return Response.json(response);
  } catch (error) {
    logger.error("Failed to create autopilot event:", error);
    return new Response("Failed to create event", { status: 500 });
  }
}

/**
 * API route for fetching autopilot events with pagination.
 * Used for loading older events when scrolling up in the session view.
 *
 * Route: /api/autopilot/sessions/:session_id/events
 *
 * Query params:
 * - limit: number of events to fetch (default 20)
 * - before: cursor for pagination (fetch events with id < before)
 */
export async function loader({ params, request }: LoaderFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  const url = new URL(request.url);
  const limitParam = url.searchParams.get("limit");
  const before = url.searchParams.get("before");

  const limit = limitParam ? parseInt(limitParam, 10) : 21;

  const client = getAutopilotClient();
  const response = await client.listAutopilotEvents(sessionId, {
    limit,
    before: before || undefined,
  });

  return Response.json({
    events: response.events,
  });
}
