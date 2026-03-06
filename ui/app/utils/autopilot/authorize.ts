import type { AuthorizationStatus } from "./types";

/**
 * Submit an authorization decision (approve or reject) for a tool call.
 * @throws Error if the request fails
 */
export async function submitToolCallAuthorization(
  sessionId: string,
  toolCallEventId: string,
  status: AuthorizationStatus,
  toolCallName: string,
  toolCallArguments: unknown,
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
        tool_call_name: toolCallName,
        tool_call_arguments: toolCallArguments,
      }),
      signal,
    },
  );
  if (!response.ok) {
    throw new Error("Authorization failed");
  }
}
