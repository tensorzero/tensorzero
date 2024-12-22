import { describe, it, expect } from "vitest";
import { content_block_to_openai_message } from "./openai";
import { create_env } from "../minijinja/pkg/minijinja_bindings";
import type {
  textInputMessageContent,
  toolCallInputMessageContent,
  toolResultInputMessageContent,
} from "../clickhouse";

describe("content_block_to_openai_message", () => {
  it("test text content block with no template", async () => {
    const textContentBlock = {
      type: "text",
      value: "foo bar baz",
    } as textInputMessageContent;
    const env = create_env({});
    const openai_message = content_block_to_openai_message(
      textContentBlock,
      "user",
      env,
    );
    expect(openai_message).toEqual({ role: "user", content: "foo bar baz" });
  });

  it("test text content block with template", async () => {
    const textContentBlock = {
      type: "text",
      value: { dogName: "Wally" },
    } as textInputMessageContent;
    const env = create_env({
      assistant: "Hi, I'm your friendly canine assistant {{ dogName }}!",
    });
    const openai_message = content_block_to_openai_message(
      textContentBlock,
      "assistant",
      env,
    );
    expect(openai_message).toEqual({
      role: "assistant",
      content: "Hi, I'm your friendly canine assistant Wally!",
    });
  });

  it("test toolCall content block", async () => {
    const toolCall = {
      type: "tool_call",
      name: "get_weather",
      arguments: '{"location": "Manaus"}',
      id: "tool_494949",
    } as toolCallInputMessageContent;
    const env = create_env({
      assistant: "Hi, I'm your friendly canine assistant {{ dogName }}!",
    });
    const openai_message = content_block_to_openai_message(
      toolCall,
      "user",
      env,
    );
    expect(openai_message).toEqual({
      role: "assistant",
      tool_calls: [
        {
          id: "tool_494949",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"location": "Manaus"}',
          },
        },
      ],
    });
  });

  it("test toolResult content block", async () => {
    const toolResult = {
      type: "tool_result",
      name: "get_weather",
      result: "35 and sunny",
      id: "tool_494949",
    } as toolResultInputMessageContent;
    const env = create_env({
      assistant: "Hi, I'm your friendly canine assistant {{ dogName }}!",
    });
    const openai_message = content_block_to_openai_message(
      toolResult,
      "user",
      env,
    );
    expect(openai_message).toEqual({
      role: "tool",
      content: "35 and sunny",
      tool_call_id: "tool_494949",
    });
  });
});
