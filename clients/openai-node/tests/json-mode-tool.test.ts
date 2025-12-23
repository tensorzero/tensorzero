/**
 * Tests for json_mode="tool" using the OpenAI Node.js client with TensorZero
 *
 * These tests verify that chat functions with json_mode="tool" properly convert
 * tool calls to text responses when using the OpenAI-compatible API, both in
 * streaming and non-streaming modes.
 */
import { describe, it, expect, beforeAll } from "vitest";
import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources";

// Client setup
let client: OpenAI;

beforeAll(() => {
  client = new OpenAI({
    apiKey: "donotuse",
    baseURL: "http://127.0.0.1:3000/openai/v1",
  });
});

describe("JSON Mode Tool", () => {
  it("should handle json_mode='tool' in non-streaming mode", async () => {
    const outputSchema = {
      type: "object",
      properties: {
        sentiment: {
          type: "string",
          enum: ["positive", "negative", "neutral"],
        },
        confidence: {
          type: "number",
        },
      },
      required: ["sentiment", "confidence"],
      additionalProperties: false,
    };

    const response_format = {
      type: "json_schema" as const,
      json_schema: {
        name: "sentiment_analysis",
        description: "Sentiment analysis schema",
        schema: outputSchema,
        strict: true,
      },
    };

    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Analyze sentiment" },
    ];

    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::test_chat_json_mode_tool_openai",
      response_format,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::params": {
        chat_completion: {
          json_mode: "tool",
        },
      },
    });

    // Verify we got a response
    expect(result.choices).toBeDefined();
    expect(result.choices.length).toBeGreaterThan(0);

    // Extract the text content
    const message = result.choices[0].message;
    expect(message.content).toBeDefined();
    expect(message.content).not.toBeNull();

    // Verify no tool_calls (should be text response)
    expect(
      message.tool_calls === null ||
        message.tool_calls === undefined ||
        message.tool_calls.length === 0
    ).toBe(true);

    // Verify the text is valid JSON
    const parsed_json = JSON.parse(message.content!);

    // Verify schema structure
    expect(parsed_json).toHaveProperty("sentiment");
    expect(parsed_json).toHaveProperty("confidence");

    // Verify the values from dummy provider
    expect(parsed_json.sentiment).toBe("positive");
    expect(parsed_json.confidence).toBe(0.95);
  });

  it("should handle json_mode='tool' in streaming mode", async () => {
    const outputSchema = {
      type: "object",
      properties: {
        sentiment: {
          type: "string",
          enum: ["positive", "negative", "neutral"],
        },
        confidence: {
          type: "number",
        },
      },
      required: ["sentiment", "confidence"],
      additionalProperties: false,
    };

    const response_format = {
      type: "json_schema" as const,
      json_schema: {
        name: "sentiment_analysis",
        description: "Sentiment analysis schema",
        schema: outputSchema,
        strict: true,
      },
    };

    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Analyze sentiment" },
    ];

    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::test_chat_json_mode_tool_openai",
      response_format,
      stream: true,
      "tensorzero::params": {
        chat_completion: {
          json_mode: "tool",
        },
      },
    });

    // Accumulate text from chunks
    let accumulated_text = "";
    let chunk_count = 0;

    for await (const chunk of stream) {
      chunk_count++;

      // Verify we're getting chat chunks
      expect(chunk.choices).toBeDefined();

      // Verify chunks are text chunks (not tool_call)
      for (const choice of chunk.choices) {
        if (
          choice.delta.content !== null &&
          choice.delta.content !== undefined
        ) {
          accumulated_text += choice.delta.content;
        }
        // Verify no tool_calls in delta
        if (choice.delta.tool_calls !== undefined) {
          expect(choice.delta.tool_calls).toHaveLength(0);
        }
      }
    }

    // Verify we got at least one chunk
    expect(chunk_count).toBeGreaterThan(0);

    // Verify the accumulated text is not empty
    expect(accumulated_text.length).toBeGreaterThan(0);

    // Verify the accumulated text is valid JSON
    const parsed_json = JSON.parse(accumulated_text);

    // Verify schema structure
    expect(parsed_json).toHaveProperty("sentiment");
    expect(parsed_json).toHaveProperty("confidence");

    // Verify the values from dummy provider
    expect(parsed_json.sentiment).toBe("positive");
    expect(parsed_json.confidence).toBe(0.95);
  });
});
