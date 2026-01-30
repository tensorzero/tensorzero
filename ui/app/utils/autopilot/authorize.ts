/**
 * Client-side API utility for authorizing a single tool call.
 */

import type { AuthorizationStatus } from "./types";

export type { AuthorizationStatus };

/**
 * Authorize (approve or reject) a single tool call.
 *
 * @param sessionId - The autopilot session ID
 * @param toolCallEventId - The tool call event ID to authorize
 * @param status - The authorization status (approved or rejected with reason)
 * @param signal - Optional AbortSignal for request cancellation
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
