import { describe, it, expect } from "vitest";
import { serializeMessages } from "./conversation-serializer";
import type {
  ContentBlockChatOutput,
  Input,
  InputMessage,
  StoredInference,
} from "~/types/tensorzero";

function makeChatInference(output?: ContentBlockChatOutput[]): StoredInference {
  return {
    type: "chat",
    function_name: "test_fn",
    variant_name: "test_variant",
    timestamp: "2024-01-01T00:00:00Z",
    episode_id: "ep_123",
    inference_id: "inf_123",
    tags: {},
    dispreferred_outputs: [],
    provider_tools: [],
    output,
  };
}

function makeJsonInference(output?: {
  raw: string | null;
  parsed: unknown;
}): StoredInference {
  return {
    type: "json",
    function_name: "test_fn",
    variant_name: "test_variant",
    timestamp: "2024-01-01T00:00:00Z",
    episode_id: "ep_123",
    inference_id: "inf_123",
    tags: {},
    dispreferred_outputs: [],
    output: output as StoredInference & { type: "json" } extends {
      output: infer O;
    }
      ? O
      : never,
  };
}

describe("serializeMessages", () => {
  it("should serialize a multi-turn chat conversation", () => {
    const input: Input = {
      system: "You are helpful.",
      messages: [
        { role: "user", content: [{ type: "text", text: "Hi" }] },
        { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
        { role: "user", content: [{ type: "text", text: "What is 2+2?" }] },
      ],
    };
    const inference = makeChatInference([{ type: "text", text: "4" }]);

    const result = JSON.parse(serializeMessages(input, inference));

    expect(result).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
      { role: "user", content: "What is 2+2?" },
      { role: "assistant", content: "4" },
    ]);
  });

  it("should serialize a tool use conversation with both tool_call variants", () => {
    const input: Input = {
      system: { assistant_name: "WeatherBot" },
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "What's the weather?" }],
        },
        {
          role: "assistant",
          content: [
            {
              // InferenceResponseToolCall variant (has raw_name/raw_arguments)
              type: "tool_call",
              id: "tc_1",
              raw_name: "get_weather",
              raw_arguments: '{"city":"SF"}',
              name: null,
              arguments: null,
            } as InputMessage["content"][number],
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              id: "tc_1",
              name: "get_weather",
              result: "Sunny, 72F",
            },
          ],
        },
      ],
    };
    const inference = makeChatInference([
      { type: "text", text: "The weather in SF is sunny and 72F." },
    ]);

    const result = JSON.parse(serializeMessages(input, inference));

    // Arguments system prompt is JSON-stringified
    expect(result[0]).toEqual({
      role: "system",
      content: JSON.stringify({ assistant_name: "WeatherBot" }),
    });
    // tool_call with null name falls back to raw_name
    expect(result[2].content).toEqual([
      {
        type: "tool_call",
        id: "tc_1",
        name: "get_weather",
        arguments: '{"city":"SF"}',
      },
    ]);
    // tool_result
    expect(result[3].content).toEqual([
      {
        type: "tool_result",
        id: "tc_1",
        name: "get_weather",
        result: "Sunny, 72F",
      },
    ]);
    // Output flattened to string (single text block)
    expect(result[4].content).toBe("The weather in SF is sunny and 72F.");
  });

  it("should serialize output with mixed block types (thought + tool_call)", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Search for it" }] },
      ],
    };
    const inference = makeChatInference([
      { type: "thought" } as ContentBlockChatOutput,
      {
        type: "tool_call",
        id: "tc_1",
        raw_name: "search",
        raw_arguments: '{"q":"test"}',
        name: "search",
        arguments: { q: "test" },
      },
    ]);

    const result = JSON.parse(serializeMessages(input, inference));

    // Multi-block output stays as array, thought with undefined text defaults to ""
    expect(result[1].content).toEqual([
      { type: "thought", text: "" },
      {
        type: "tool_call",
        id: "tc_1",
        name: "search",
        arguments: '{"q":"test"}',
      },
    ]);
  });

  it("should serialize JSON inference preferring raw over parsed", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Extract data" }] },
      ],
    };

    // raw present → use raw
    const withRaw = makeJsonInference({
      raw: '{"name": "John"}',
      parsed: { name: "John" },
    });
    const result1 = JSON.parse(serializeMessages(input, withRaw));
    expect(result1[1].content).toBe('{"name": "John"}');

    // raw null → JSON.stringify(parsed)
    const withoutRaw = makeJsonInference({
      raw: null,
      parsed: { name: "John" },
    });
    const result2 = JSON.parse(serializeMessages(input, withoutRaw));
    expect(result2[1].content).toBe(JSON.stringify({ name: "John" }));
  });

  it("should handle edge cases: undefined input, no output, empty messages", () => {
    // Undefined input + output → output only
    const outputOnly = makeChatInference([
      { type: "text", text: "I can help." },
    ]);
    expect(JSON.parse(serializeMessages(undefined, outputOnly))).toEqual([
      { role: "assistant", content: "I can help." },
    ]);

    // Undefined input + no output → empty array
    const nothing = makeChatInference(undefined);
    expect(JSON.parse(serializeMessages(undefined, nothing))).toEqual([]);

    // No system prompt → no system message
    const noSystem: Input = {
      messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
    };
    const result = JSON.parse(serializeMessages(noSystem, makeChatInference()));
    expect(result[0].role).toBe("user");
  });
});
