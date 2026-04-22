import { describe, it, expect, vi } from "vitest";
import type {
  GatewayEvent,
  GatewayListConfigWritesResponse,
} from "~/types/tensorzero";
import {
  extractEditPayloadsFromConfigWrite,
  listAllConfigWrites,
} from "./autopilot-client";

function makeEvent(id: string): GatewayEvent {
  return {
    id,
    session_id: "session-1",
    created_at: "2025-01-01T00:00:00Z",
    payload: {
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: `event-${id}` }],
      metadata: {},
    },
  };
}

function makeMockClient(pages: GatewayEvent[][]) {
  let callIndex = 0;
  const listConfigWrites = vi.fn(
    async (): Promise<GatewayListConfigWritesResponse> => {
      const page = pages[callIndex] ?? [];
      callIndex++;
      return { config_writes: page };
    },
  );
  return { listConfigWrites };
}

const EDIT_ARGS = {
  config: {},
  extra_templates: {},
  edit: [
    {
      operation: "upsert_variant" as const,
      function_name: "fn",
      variant_name: "v",
      variant: { type: "chat_completion" as const },
    },
  ],
};

describe("extractEditPayloadsFromConfigWrite", () => {
  it("should extract edits from a tool_call event", () => {
    const event: GatewayEvent = {
      id: "evt-1",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "tool_call",
        name: "write_config",
        arguments: EDIT_ARGS,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        side_info: {} as any,
        requires_approval: false,
      },
    };
    const result = extractEditPayloadsFromConfigWrite(event);
    expect(result).toEqual(EDIT_ARGS.edit);
  });

  it("should extract edits from a tool_result event", () => {
    const event: GatewayEvent = {
      id: "evt-2",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "tool_result",
        tool_call_event_id: "evt-1",
        outcome: { type: "success", result: "ok" },
        tool_call_name: "write_config",
        tool_call_arguments: EDIT_ARGS,
        tool_call_authorization_source: { type: "automatic" },
        tool_call_authorization_status: { type: "approved" },
      },
    };
    const result = extractEditPayloadsFromConfigWrite(event);
    expect(result).toEqual(EDIT_ARGS.edit);
  });

  it("should extract edits from a tool_call_authorization event", () => {
    const event: GatewayEvent = {
      id: "evt-3",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "tool_call_authorization",
        source: { type: "ui" },
        tool_call_event_id: "evt-1",
        status: { type: "approved" },
        tool_call_name: "write_config",
        tool_call_arguments: EDIT_ARGS,
      },
    };
    const result = extractEditPayloadsFromConfigWrite(event);
    expect(result).toEqual(EDIT_ARGS.edit);
  });

  it("should throw for non-write_config tool_call", () => {
    const event: GatewayEvent = {
      id: "evt-4",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "tool_call",
        name: "other_tool",
        arguments: {},
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        side_info: {} as any,
        requires_approval: false,
      },
    };
    expect(() => extractEditPayloadsFromConfigWrite(event)).toThrow(
      "Expected write_config tool call but got other_tool",
    );
  });

  it("should throw for non-tool event types", () => {
    const event: GatewayEvent = {
      id: "evt-5",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
        metadata: {},
      },
    };
    expect(() => extractEditPayloadsFromConfigWrite(event)).toThrow(
      "Expected tool_call, tool_result, or tool_call_authorization",
    );
  });

  it("should throw when edit payload is missing", () => {
    const event: GatewayEvent = {
      id: "evt-6",
      session_id: "s-1",
      created_at: "2025-01-01T00:00:00Z",
      payload: {
        type: "tool_call",
        name: "write_config",
        arguments: { config: {}, extra_templates: {} },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        side_info: {} as any,
        requires_approval: false,
      },
    };
    expect(() => extractEditPayloadsFromConfigWrite(event)).toThrow(
      "does not have an edit payload",
    );
  });
});

describe("listAllConfigWrites", () => {
  it("should return empty array when there are no config writes", async () => {
    const client = makeMockClient([[]]);
    const result = await listAllConfigWrites(client, "session-1", 3);
    expect(result, "should return empty array for no config writes").toEqual(
      [],
    );
    expect(
      client.listConfigWrites,
      "should call listConfigWrites exactly once",
    ).toHaveBeenCalledTimes(1);
  });

  it("should return all items when they fit in a single page", async () => {
    const events = [makeEvent("1"), makeEvent("2")];
    // With pageSize=3, we request 4 items. Getting 2 back means no more pages.
    const client = makeMockClient([events]);
    const result = await listAllConfigWrites(client, "session-1", 3);
    expect(result, "should return all events from single page").toEqual(events);
    expect(
      client.listConfigWrites,
      "should call listConfigWrites exactly once",
    ).toHaveBeenCalledTimes(1);
    expect(
      client.listConfigWrites,
      "should request pageSize + 1 items",
    ).toHaveBeenCalledWith("session-1", { limit: 4, offset: 0 });
  });

  it("should paginate when results exceed page size", async () => {
    const page1 = [
      makeEvent("1"),
      makeEvent("2"),
      makeEvent("3"),
      makeEvent("4"),
    ]; // 4 items = pageSize+1, means more pages
    const page2 = [makeEvent("5")]; // less than pageSize+1, last page

    const client = makeMockClient([page1, page2]);
    const result = await listAllConfigWrites(client, "session-1", 3);

    expect(result, "should return all events across pages").toEqual([
      makeEvent("1"),
      makeEvent("2"),
      makeEvent("3"),
      makeEvent("5"),
    ]);
    expect(
      client.listConfigWrites,
      "should call listConfigWrites twice for two pages",
    ).toHaveBeenCalledTimes(2);
    expect(client.listConfigWrites).toHaveBeenNthCalledWith(1, "session-1", {
      limit: 4,
      offset: 0,
    });
    expect(client.listConfigWrites).toHaveBeenNthCalledWith(2, "session-1", {
      limit: 4,
      offset: 3,
    });
  });

  it("should handle exactly pageSize items (no extra item)", async () => {
    // Exactly 3 items returned when we requested 4 — no more pages
    const events = [makeEvent("1"), makeEvent("2"), makeEvent("3")];
    const client = makeMockClient([events]);
    const result = await listAllConfigWrites(client, "session-1", 3);
    expect(
      result,
      "should return all events when count equals page size",
    ).toEqual(events);
    expect(
      client.listConfigWrites,
      "should only make one call when results fit exactly in page",
    ).toHaveBeenCalledTimes(1);
  });

  it("should paginate across three pages", async () => {
    const page1 = [makeEvent("1"), makeEvent("2"), makeEvent("3")]; // 3 = pageSize+1
    const page2 = [makeEvent("4"), makeEvent("5"), makeEvent("6")]; // 3 = pageSize+1
    const page3 = [makeEvent("7")]; // < pageSize+1, last page

    const client = makeMockClient([page1, page2, page3]);
    const result = await listAllConfigWrites(client, "session-1", 2);

    expect(result, "should return all 5 events across three pages").toEqual([
      makeEvent("1"),
      makeEvent("2"),
      makeEvent("4"),
      makeEvent("5"),
      makeEvent("7"),
    ]);
    expect(
      client.listConfigWrites,
      "should make three calls for three pages",
    ).toHaveBeenCalledTimes(3);
    expect(client.listConfigWrites).toHaveBeenNthCalledWith(1, "session-1", {
      limit: 3,
      offset: 0,
    });
    expect(client.listConfigWrites).toHaveBeenNthCalledWith(2, "session-1", {
      limit: 3,
      offset: 2,
    });
    expect(client.listConfigWrites).toHaveBeenNthCalledWith(3, "session-1", {
      limit: 3,
      offset: 4,
    });
  });
});
