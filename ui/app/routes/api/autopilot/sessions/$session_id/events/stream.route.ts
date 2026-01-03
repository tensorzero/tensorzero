import type { LoaderFunctionArgs } from "react-router";
import { getEnv } from "~/utils/env.server";

/**
 * Streaming API route that proxies SSE events from the TensorZero Gateway.
 * This allows the client to stream autopilot events without exposing the gateway URL.
 *
 * Route: /api/autopilot/sessions/:session_id/events/stream
 */
export async function loader({ params, request }: LoaderFunctionArgs) {
  const sessionId = params.session_id;
  if (!sessionId) {
    return new Response("Session ID is required", { status: 400 });
  }

  const url = new URL(request.url);
  const lastEventId = url.searchParams.get("last_event_id");

  const env = getEnv();
  const gatewayUrl = new URL(
    `/internal/autopilot/v1/sessions/${encodeURIComponent(sessionId)}/events/stream`,
    env.TENSORZERO_GATEWAY_URL,
  );

  if (lastEventId) {
    gatewayUrl.searchParams.set("last_event_id", lastEventId);
  }

  const headers: Record<string, string> = {
    Accept: "text/event-stream",
  };

  if (env.TENSORZERO_API_KEY) {
    headers["Authorization"] = `Bearer ${env.TENSORZERO_API_KEY}`;
  }

  try {
    const response = await fetch(gatewayUrl.toString(), {
      headers,
      signal: request.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      return new Response(errorText, { status: response.status });
    }

    // Pass through the SSE stream
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to connect to gateway";
    return new Response(message, { status: 502 });
  }
}
