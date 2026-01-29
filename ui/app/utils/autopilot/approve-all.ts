/**
 * Client-side API utilities for Autopilot operations.
 */

/**
 * Batch-approve all pending tool calls up to and including the specified event ID.
 *
 * @param sessionId - The autopilot session ID
 * @param lastToolCallEventId - Approve all pending tool calls with IDs <= this value
 * @param signal - Optional AbortSignal for request cancellation
 * @throws Error if the request fails
 */
export async function approveAllToolCalls(
  sessionId: string,
  lastToolCallEventId: string,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `/api/autopilot/sessions/${encodeURIComponent(sessionId)}/actions/approve_all`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        last_tool_call_event_id: lastToolCallEventId,
      }),
      signal,
    },
  );
  if (!response.ok) {
    throw new Error("Batch approval failed");
  }
}
