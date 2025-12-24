import { BaseTensorZeroClient } from "./base-client";
import type {
  Event,
  CreateEventRequest,
  CreateEventResponse,
  ListEventsParams,
  ListEventsResponse,
  ListSessionsParams,
  ListSessionsResponse,
  StreamEventsParams,
} from "~/types/tensorzero";

/**
 * A client for calling TensorZero Autopilot API endpoints.
 */
export class AutopilotClient extends BaseTensorZeroClient {
  /**
   * Lists autopilot sessions with optional pagination.
   */
  async listAutopilotSessions(
    params?: ListSessionsParams,
  ): Promise<ListSessionsResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set("limit", params.limit.toString());
    if (params?.offset) searchParams.set("offset", params.offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/autopilot/v1/sessions${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListSessionsResponse;
  }

  /**
   * Lists events for an autopilot session with optional pagination.
   */
  async listAutopilotEvents(
    sessionId: string,
    params?: ListEventsParams,
  ): Promise<ListEventsResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set("limit", params.limit.toString());
    if (params?.before) searchParams.set("before", params.before);
    const queryString = searchParams.toString();
    const endpoint = `/internal/autopilot/v1/sessions/${encodeURIComponent(sessionId)}/events${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListEventsResponse;
  }

  /**
   * Creates an event in an autopilot session.
   * Use session_id = "00000000-0000-0000-0000-000000000000" (nil UUID) to create a new session.
   */
  async createAutopilotEvent(
    sessionId: string,
    request: CreateEventRequest,
  ): Promise<CreateEventResponse> {
    const endpoint = `/internal/autopilot/v1/sessions/${encodeURIComponent(sessionId)}/events`;
    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CreateEventResponse;
  }

  /**
   * Streams events for an autopilot session using Server-Sent Events.
   * Returns an async generator that yields Event objects.
   */
  async *streamAutopilotEvents(
    sessionId: string,
    params?: StreamEventsParams,
  ): AsyncGenerator<Event, void, unknown> {
    const searchParams = new URLSearchParams();
    if (params?.last_event_id)
      searchParams.set("last_event_id", params.last_event_id);
    const queryString = searchParams.toString();
    const url = `${this.baseUrl}/internal/autopilot/v1/sessions/${encodeURIComponent(sessionId)}/events/stream${queryString ? `?${queryString}` : ""}`;

    const headers: Record<string, string> = {
      Accept: "text/event-stream",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(url, { headers });
    if (!response.ok) {
      const message = await response.text();
      this.handleHttpError({ message, response });
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data) {
              const event = JSON.parse(data) as Event;
              yield event;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
