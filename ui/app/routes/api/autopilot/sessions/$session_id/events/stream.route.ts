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

    // Wrap the stream to handle client disconnection gracefully.
    // Without this, abort errors propagate as unhandled exceptions when
    // the client navigates away mid-stream.
    const wrappedStream = new ReadableStream({
      async start(controller) {
        const reader = response.body?.getReader();
        if (!reader) {
          controller.close();
          return;
        }

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
          controller.close();
        } catch {
          // Client disconnected - close gracefully
          controller.close();
        } finally {
          reader.releaseLock();
        }
      },
    });

    return new Response(wrappedStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    // Client disconnected - return silently since they won't see the response anyway
    if (error instanceof Error && error.name === "AbortError") {
      return new Response();
    }

    const message =
      error instanceof Error ? error.message : "Failed to connect to gateway";
    return new Response(message, { status: 502 });
  }
}
