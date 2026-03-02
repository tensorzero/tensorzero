import { describe, it, expect } from "vitest";
import { serializeConversation } from "./conversation-serializer";
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

describe("serializeConversation", () => {
  it("should serialize a basic chat conversation with string system prompt", () => {
    const input: Input = {
      system: "You are a helpful assistant.",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    };
    const inference = makeChatInference([{ type: "text", text: "Hi there!" }]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
    ]);
  });

  it("should serialize an Arguments system prompt as JSON", () => {
    const input: Input = {
      system: { tone: "formal", language: "en" },
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    };
    const inference = makeChatInference([{ type: "text", text: "Greetings." }]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0]).toEqual({
      role: "system",
      content: JSON.stringify({ tone: "formal", language: "en" }),
    });
  });

  it("should handle missing system prompt", () => {
    const input: Input = {
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    };
    const inference = makeChatInference([{ type: "text", text: "Hi!" }]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi!" },
    ]);
  });

  it("should flatten single text content blocks to a plain string", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "One block" }] },
      ],
    };
    const inference = makeChatInference([
      { type: "text", text: "Single output" },
    ]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toBe("One block");
    expect(result[1].content).toBe("Single output");
  });

  it("should keep multiple content blocks as an array", () => {
    const input: Input = {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "First" },
            { type: "text", text: "Second" },
          ],
        },
      ],
    };
    const inference = makeChatInference([]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      { type: "text", text: "First" },
      { type: "text", text: "Second" },
    ]);
  });

  it("should serialize tool_call content blocks (ToolCall variant)", () => {
    const input: Input = {
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              id: "tc_1",
              name: "get_weather",
              arguments: '{"city":"SF"}',
            },
          ],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      {
        type: "tool_call",
        id: "tc_1",
        name: "get_weather",
        arguments: '{"city":"SF"}',
      },
    ]);
  });

  it("should serialize tool_call content blocks (InferenceResponseToolCall variant)", () => {
    const input: Input = {
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "tool_call",
              id: "tc_1",
              raw_name: "get_weather",
              raw_arguments: '{"city":"SF"}',
              name: "get_weather",
              arguments: { city: "SF" },
            } as InputMessage["content"][number],
          ],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      {
        type: "tool_call",
        id: "tc_1",
        name: "get_weather",
        arguments: '{"city":"SF"}',
      },
    ]);
  });

  it("should serialize tool_result content blocks", () => {
    const input: Input = {
      messages: [
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
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      {
        type: "tool_result",
        id: "tc_1",
        name: "get_weather",
        result: "Sunny, 72F",
      },
    ]);
  });

  it("should serialize thought content blocks", () => {
    const input: Input = {
      messages: [
        {
          role: "assistant",
          content: [{ type: "thought", text: "Let me think..." }],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      { type: "thought", text: "Let me think..." },
    ]);
  });

  it("should serialize template content blocks as text with stringified arguments", () => {
    const input: Input = {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "template",
              name: "greeting",
              arguments: { name: "World" },
            },
          ],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    // Single text block gets flattened to a plain string
    expect(result[0].content).toBe(JSON.stringify({ name: "World" }));
  });

  it("should serialize raw_text as text", () => {
    const input: Input = {
      messages: [
        {
          role: "user",
          content: [{ type: "raw_text", value: "Raw content here" }],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toBe("Raw content here");
  });

  it("should serialize file content blocks with file_type only", () => {
    const input: Input = {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "file",
              file_type: "url",
              url: "https://example.com/image.png",
              mime_type: "image/png",
            } as InputMessage["content"][number],
          ],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([{ type: "file", file_type: "url" }]);
  });

  it("should serialize unknown content blocks", () => {
    const input: Input = {
      messages: [
        {
          role: "user",
          content: [{ type: "unknown", data: { custom: "data" } }],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      { type: "unknown", data: { custom: "data" } },
    ]);
  });

  it("should handle chat inference with no output", () => {
    const input: Input = {
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    };
    const inference = makeChatInference(undefined);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([{ role: "user", content: "Hello" }]);
  });

  it("should serialize chat output tool_call blocks", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Get weather" }] },
      ],
    };
    const inference = makeChatInference([
      {
        type: "tool_call",
        id: "tc_1",
        raw_name: "get_weather",
        raw_arguments: '{"city":"NYC"}',
        name: "get_weather",
        arguments: { city: "NYC" },
      },
    ]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[1]).toEqual({
      role: "assistant",
      content: [
        {
          type: "tool_call",
          id: "tc_1",
          name: "get_weather",
          arguments: '{"city":"NYC"}',
        },
      ],
    });
  });

  it("should serialize JSON inference with raw output", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Extract data" }] },
      ],
    };
    const inference = makeJsonInference({
      raw: '{"name": "John"}',
      parsed: { name: "John" },
    });

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[1]).toEqual({
      role: "assistant",
      content: '{"name": "John"}',
    });
  });

  it("should serialize JSON inference with null raw but present parsed", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Extract data" }] },
      ],
    };
    const inference = makeJsonInference({
      raw: null,
      parsed: { name: "John" },
    });

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[1]).toEqual({
      role: "assistant",
      content: JSON.stringify({ name: "John" }),
    });
  });

  it("should handle undefined input and still serialize output", () => {
    const inference = makeChatInference([
      { type: "text", text: "I can help with that." },
    ]);

    const result = JSON.parse(serializeConversation(undefined, inference));

    expect(result).toEqual([
      { role: "assistant", content: "I can help with that." },
    ]);
  });

  it("should handle a multi-turn conversation", () => {
    const input: Input = {
      system: "You are helpful.",
      messages: [
        { role: "user", content: [{ type: "text", text: "Hi" }] },
        { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
        { role: "user", content: [{ type: "text", text: "What is 2+2?" }] },
      ],
    };
    const inference = makeChatInference([{ type: "text", text: "4" }]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
      { role: "user", content: "What is 2+2?" },
      { role: "assistant", content: "4" },
    ]);
  });

  it("should handle tool_call with null name falling back to raw_name", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Get weather" }] },
      ],
    };
    const inference = makeChatInference([
      {
        type: "tool_call",
        id: "tc_1",
        raw_name: "get_weather",
        raw_arguments: '{"city":"NYC"}',
        name: null,
        arguments: null,
      },
    ]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[1].content).toEqual([
      {
        type: "tool_call",
        id: "tc_1",
        name: "get_weather",
        arguments: '{"city":"NYC"}',
      },
    ]);
  });

  it("should handle thought with undefined text", () => {
    const input: Input = {
      messages: [
        {
          role: "assistant",
          content: [{ type: "thought" } as InputMessage["content"][number]],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([{ type: "thought", text: "" }]);
  });

  it("should handle JSON inference with undefined output", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Extract data" }] },
      ],
    };
    const inference = makeJsonInference(undefined);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([{ role: "user", content: "Extract data" }]);
  });

  it("should handle empty messages array", () => {
    const input: Input = {
      system: "You are helpful.",
      messages: [],
    };
    const inference = makeChatInference([{ type: "text", text: "Hi!" }]);

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "assistant", content: "Hi!" },
    ]);
  });

  it("should handle mixed content block types in a single message", () => {
    const input: Input = {
      messages: [
        {
          role: "assistant",
          content: [
            { type: "thought", text: "Thinking..." },
            { type: "text", text: "Here is the answer." },
            {
              type: "tool_call",
              id: "tc_1",
              name: "search",
              arguments: '{"q":"test"}',
            },
          ],
        },
      ],
    };
    const inference = makeChatInference();

    const result = JSON.parse(serializeConversation(input, inference));

    expect(result[0].content).toEqual([
      { type: "thought", text: "Thinking..." },
      { type: "text", text: "Here is the answer." },
      {
        type: "tool_call",
        id: "tc_1",
        name: "search",
        arguments: '{"q":"test"}',
      },
    ]);
  });

  it("should return valid JSON string with proper formatting", () => {
    const input: Input = {
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
    };
    const inference = makeChatInference([{ type: "text", text: "Hi!" }]);

    const result = serializeConversation(input, inference);

    expect(result).toContain("\n");
    expect(() => JSON.parse(result)).not.toThrow();
  });
});
