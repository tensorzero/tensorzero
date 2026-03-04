import { expect, test, describe } from "vitest";
import type { GatewayEvent } from "~/types/tensorzero";
import { detectEventVisualization } from "./EventStream";

function makeEvent(payload: GatewayEvent["payload"]): GatewayEvent {
  return {
    id: "evt-1",
    payload,
    session_id: "sess-1",
    created_at: "2025-01-01T00:00:00Z",
  };
}

describe("detectEventVisualization", () => {
  test("returns top_k_evaluation for visualization event with known type", () => {
    const topKData = {
      type: "top_k_evaluation" as const,
      metric_name: "accuracy",
      function_name: "my_fn",
      k: 5,
      evaluations: [],
    };
    const event = makeEvent({
      type: "visualization",
      tool_execution_id: "tool-1",
      visualization: topKData,
    });
    const result = detectEventVisualization(event);
    expect(result).toEqual({ type: "top_k_evaluation", data: topKData });
  });

  test("returns unknown for visualization event with unrecognized type", () => {
    const unknownViz = { type: "some_future_viz", foo: "bar" };
    const event = makeEvent({
      type: "visualization",
      tool_execution_id: "tool-1",
      visualization: unknownViz,
    });
    const result = detectEventVisualization(event);
    expect(result).toEqual({ type: "unknown", raw: unknownViz });
  });

  test("returns unknown for visualization event with non-object payload", () => {
    const event = makeEvent({
      type: "visualization",
      tool_execution_id: "tool-1",
      visualization: "just a string" as never,
    });
    const result = detectEventVisualization(event);
    expect(result).toEqual({ type: "unknown", raw: "just a string" });
  });

  test("returns feedback_by_variant for tool_result with valid feedback data", () => {
    const feedbackData = [
      { variant_name: "v1", mean: 0.8, variance: 0.04, count: 100 },
      { variant_name: "v2", mean: 0.6, variance: null, count: 50 },
    ];
    const event = makeEvent({
      type: "tool_result",
      tool_call_event_id: "tc-1",
      outcome: {
        type: "success",
        result: JSON.stringify(feedbackData),
      },
    });
    const result = detectEventVisualization(event);
    expect(result).toEqual({ type: "feedback_by_variant", data: feedbackData });
  });

  test("returns null for tool_result with non-feedback data", () => {
    const event = makeEvent({
      type: "tool_result",
      tool_call_event_id: "tc-1",
      outcome: {
        type: "success",
        result: JSON.stringify({ some: "other data" }),
      },
    });
    expect(detectEventVisualization(event)).toBeNull();
  });

  test("returns null for tool_result with failure outcome", () => {
    const event = makeEvent({
      type: "tool_result",
      tool_call_event_id: "tc-1",
      outcome: {
        type: "failure",
        error: { message: "something broke" },
      },
    });
    expect(detectEventVisualization(event)).toBeNull();
  });

  test("returns null for message events", () => {
    const event = makeEvent({
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "hello" }],
      metadata: {},
    });
    expect(detectEventVisualization(event)).toBeNull();
  });

  test("returns null for tool_call events", () => {
    const event = makeEvent({
      type: "tool_call",
      name: "get_feedback_by_variant",
      arguments: {},
      side_info: {
        tool_call_event_id: "tc-1",
        session_id: "sess-1",
        config_snapshot_hash: "abc",
        optimization: {
          poll_interval_secs: BigInt(60),
          max_wait_secs: BigInt(86400),
        },
      },
    });
    expect(detectEventVisualization(event)).toBeNull();
  });

  test("returns null for tool_result with invalid JSON", () => {
    const event = makeEvent({
      type: "tool_result",
      tool_call_event_id: "tc-1",
      outcome: {
        type: "success",
        result: "not valid json {{{",
      },
    });
    expect(detectEventVisualization(event)).toBeNull();
  });

  test("returns null for tool_result with empty feedback array", () => {
    const event = makeEvent({
      type: "tool_result",
      tool_call_event_id: "tc-1",
      outcome: {
        type: "success",
        result: JSON.stringify([]),
      },
    });
    expect(detectEventVisualization(event)).toBeNull();
  });
});
