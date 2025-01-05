import { describe, it, expect } from "vitest";
import {
  content_block_to_openai_message,
  tensorzero_inference_to_openai_messages,
} from "./openai";
import { create_env } from "../minijinja/pkg/minijinja_bindings";
import type {
  ParsedInferenceRow,
  textInputMessageContent,
  toolCallInputMessageContent,
  toolResultInputMessageContent,
} from "../clickhouse/common";

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

describe("tensorzero_inference_to_openai_messages", async () => {
  it("test simple json", () => {
    const env = create_env({
      system: "Do NER Properly!",
    });
    const row = {
      variant_name: "turbo",
      input: {
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                value:
                  'We have in no way seen any Iraqi troops in the city or in its approaches , " a U.N. relief official told Reuters .',
              },
            ],
          },
        ],
      },
      output: {
        raw: '{"person":[],"organization":["U.N.","Reuters"],"location":[],"miscellaneous":["Iraqi"]}',
        parsed: {
          person: [],
          organization: ["U.N.", "Reuters"],
          location: [],
          miscellaneous: ["Iraqi"],
        },
      },
      episode_id: "0192ced0-a2c6-7323-be23-ce4124e683d3",
    } as ParsedInferenceRow;

    const openaiMessages = tensorzero_inference_to_openai_messages(row, env);
    expect(openaiMessages.length).toBe(3);
    expect(openaiMessages[0]).toStrictEqual({
      role: "system",
      content: "Do NER Properly!",
    });
    expect(openaiMessages[1]).toStrictEqual({
      role: "user",
      content:
        'We have in no way seen any Iraqi troops in the city or in its approaches , " a U.N. relief official told Reuters .',
    });
    expect(openaiMessages[2]).toStrictEqual({
      role: "assistant",
      content:
        '{"person":[],"organization":["U.N.","Reuters"],"location":[],"miscellaneous":["Iraqi"]}',
    });
  });
  it("test chat with tool calls", () => {
    const env = create_env({
      system: "Help me out with the weather!",
    });
    const row = {
      variant_name: "turbo",
      input: {
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                value: "What is the weather in Columbus?",
              },
            ],
          },
          {
            role: "assistant",
            content: [
              {
                type: "tool_call",
                arguments: { location: "Columbus" },
                id: "tool3",
                name: "get_weather",
                raw_arguments: '{"location", "Columbus"}',
                raw_name: "get_weather",
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                name: "get_weather",
                result: "34F, clear skies",
                id: "tool3",
              },
            ],
          },
        ],
      },
      output: [{ type: "text", text: "it is 34 and sunny" }],
      episode_id: "0192ced0-a2c6-7323-be23-ce4124e683d3",
    } as ParsedInferenceRow;

    const openaiMessages = tensorzero_inference_to_openai_messages(row, env);
    expect(openaiMessages.length).toBe(5);
    expect(openaiMessages[0]).toStrictEqual({
      role: "system",
      content: "Help me out with the weather!",
    });
    expect(openaiMessages[1]).toStrictEqual({
      role: "user",
      content: "What is the weather in Columbus?",
    });
    expect(openaiMessages[2]).toStrictEqual({
      role: "assistant",
      tool_calls: [
        {
          id: "tool3",
          type: "function",
          function: {
            name: "get_weather",
            // TODO: double check that this should not be serialized
            arguments: { location: "Columbus" },
          },
        },
      ],
    });
    expect(openaiMessages[3]).toStrictEqual({
      role: "tool",
      content: "34F, clear skies",
      tool_call_id: "tool3",
    });
    expect(openaiMessages[4]).toStrictEqual({
      role: "assistant",
      content: "it is 34 and sunny",
    });
  });
});
