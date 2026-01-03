import type { LoaderFunctionArgs } from "react-router";
import { getAutopilotClient } from "~/utils/tensorzero.server";

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
