import type { AuthorizationStatus } from "./types";

/**
 * Authorize (approve or reject) a single tool call.
 * @throws Error if the request fails
 */
export async function authorizeToolCall(
  sessionId: string,
  toolCallEventId: string,
  status: AuthorizationStatus,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/events/authorize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_call_event_id: toolCallEventId,
        status,
      }),
      signal,
    },
  );
  if (!response.ok) {
    throw new Error("Authorization failed");
  }
}
