/**
 * Tests for OpenAI Responses API integration
 */
import { describe, it, expect, beforeAll } from "vitest";
import OpenAI from "openai";
import {
  ChatCompletionMessageParam,
  ChatCompletionMessageFunctionToolCall,
} from "openai/resources";
import { v7 as uuidv7 } from "uuid";

// Client setup
let client: OpenAI;

beforeAll(() => {
  client = new OpenAI({
    apiKey: "donotuse",
    baseURL: "http://127.0.0.1:3000/openai/v1",
  });
});

describe("OpenAI Responses API", () => {
  it.concurrent(
    "should perform basic inference using OpenAI Responses API",
    async () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content: "What is 2+2?",
        },
      ];

      const episodeId = uuidv7();
      const result = await client.chat.completions.create({
        messages,
        model: "tensorzero::model_name::responses-gpt-5-mini",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      });

      // The response should contain content
      expect(result.choices[0].message.content).not.toBeNull();
      expect(result.choices[0].message.content!.length).toBeGreaterThan(0);

      // Extract the text content because the response might include reasoning and more
      // In OpenAI API, content is a single string, not separate blocks like TensorZero SDK
      expect(result.choices[0].message.content).toContain("4");

      expect(result.usage).not.toBeNull();
      expect(result.usage?.prompt_tokens).toBeGreaterThan(0);
      expect(result.usage?.completion_tokens).toBeGreaterThan(0);
      // TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
      // expect(result.choices[0].finish_reason).toBe("stop");
    }
  );

  it.concurrent(
    "should perform basic inference using OpenAI Responses API (streaming)",
    async () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content: "What is 2+2?",
        },
      ];

      const episodeId = uuidv7();
      // @ts-expect-error - custom TensorZero property
      const stream = await client.chat.completions.create({
        messages,
        model: "tensorzero::model_name::responses-gpt-5-mini",
        stream: true,
        stream_options: { include_usage: true },
        "tensorzero::episode_id": episodeId,
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);

      // Verify consistency across chunks
      let previousInferenceId: string | null = null;
      let previousEpisodeId: string | null = null;
      const textChunks: string[] = [];

      for (const chunk of chunks) {
        if (previousInferenceId !== null) {
          expect(chunk.id).toBe(previousInferenceId);
        }
        if (previousEpisodeId !== null) {
          // @ts-expect-error - custom TensorZero property
          expect(chunk.episode_id).toBe(previousEpisodeId);
        }
        previousInferenceId = chunk.id;
        // @ts-expect-error - custom TensorZero property
        previousEpisodeId = chunk.episode_id;

        // Collect text chunks (all chunks except the final usage-only chunk)
        if (chunk.choices && chunk.choices[0]?.delta?.content) {
          textChunks.push(chunk.choices[0].delta.content);
        }
      }

      // Should have received text content with "4" in it
      expect(textChunks.length).toBeGreaterThan(0);
      const fullText = textChunks.join("");
      expect(fullText).toContain("4");

      // Last chunk should have usage
      expect(chunks[chunks.length - 1].usage).not.toBeNull();
      expect(chunks[chunks.length - 1].usage?.prompt_tokens).toBeGreaterThan(0);
      expect(
        chunks[chunks.length - 1].usage?.completion_tokens
      ).toBeGreaterThan(0);

      // TODO (#4041): Check `finish_reason` when we improve handling of `incomplete_details.reason`.
    }
  );

  it.concurrent(
    "should handle web search",
    async () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content: "What is the current population of Tokyo?",
        },
      ];

      const episodeId = uuidv7();
      const result = await client.chat.completions.create({
        messages,
        model: "tensorzero::model_name::responses-gpt-5-mini-web-search",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      });

      // The response should contain content
      expect(result.choices[0].message.content).not.toBeNull();
      expect(result.choices[0].message.content!.length).toBeGreaterThan(0);

      // Check that web search actually happened by looking for citations in markdown format
      expect(result.choices[0].message.content).toContain("](");

      // TODO (#4042): Check for web_search_call content blocks when we expose them in the OpenAI API
      // The TensorZero SDK returns web_search_call content blocks, but the OpenAI API doesn't expose them yet

      expect(result.usage).not.toBeNull();
      expect(result.usage?.prompt_tokens).toBeGreaterThan(0);
      expect(result.usage?.completion_tokens).toBeGreaterThan(0);
    },
    90000
  );

  it.concurrent(
    "should handle web search (streaming)",
    async () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content: "What is the current population of Tokyo?",
        },
      ];

      const episodeId = uuidv7();
      // @ts-expect-error - custom TensorZero property
      const stream = await client.chat.completions.create({
        messages,
        model: "tensorzero::model_name::responses-gpt-5-mini-web-search",
        stream: true,
        stream_options: { include_usage: true },
        "tensorzero::episode_id": episodeId,
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);

      // Verify consistency across chunks and collect text
      let previousInferenceId: string | null = null;
      let previousEpisodeId: string | null = null;
      const textChunks: string[] = [];

      for (const chunk of chunks) {
        if (previousInferenceId !== null) {
          expect(chunk.id).toBe(previousInferenceId);
        }
        if (previousEpisodeId !== null) {
          // @ts-expect-error - custom TensorZero property
          expect(chunk.episode_id).toBe(previousEpisodeId);
        }
        previousInferenceId = chunk.id;
        // @ts-expect-error - custom TensorZero property
        previousEpisodeId = chunk.episode_id;

        // Collect text chunks
        if (chunk.choices && chunk.choices[0]?.delta?.content) {
          textChunks.push(chunk.choices[0].delta.content);
        }
      }

      // Last chunk should have usage
      expect(chunks[chunks.length - 1].usage).not.toBeNull();
      expect(chunks[chunks.length - 1].usage?.prompt_tokens).toBeGreaterThan(0);
      expect(
        chunks[chunks.length - 1].usage?.completion_tokens
      ).toBeGreaterThan(0);

      // Check that web search actually happened by looking for citations in markdown format
      const fullText = textChunks.join("");
      expect(fullText).toContain("](");

      // TODO (#4044): check for unknown web search events when we start returning them
    },
    90000
  );

  it.concurrent("should handle tool calls", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: "What's the temperature in Tokyo in Celsius?",
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::responses-gpt-5-mini",
      tools: [
        {
          type: "function",
          function: {
            name: "get_temperature",
            description: "Get the current temperature in a given location",
            parameters: {
              type: "object",
              properties: {
                location: {
                  type: "string",
                  description:
                    'The location to get the temperature for (e.g. "New York")',
                },
                units: {
                  type: "string",
                  description:
                    'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                  enum: ["fahrenheit", "celsius"],
                },
              },
              required: ["location"],
              additionalProperties: false,
            },
          },
        },
      ],
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    // The response should contain a tool call
    expect(result.choices[0].message.tool_calls).not.toBeNull();
    expect(result.choices[0].message.tool_calls?.length).toBeGreaterThan(0);

    const toolCall = result.choices[0].message
      .tool_calls![0] as ChatCompletionMessageFunctionToolCall;
    expect(toolCall.function.name).toBe("get_temperature");
    expect(toolCall.function.arguments).not.toBeNull();
    expect(toolCall.function.arguments).toContain("location");

    expect(result.usage).not.toBeNull();
    expect(result.usage?.prompt_tokens).toBeGreaterThan(0);
    expect(result.usage?.completion_tokens).toBeGreaterThan(0);
  });

  it.concurrent("should handle tool calls (streaming)", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: "What's the temperature in Tokyo in Celsius?",
      },
    ];

    const episodeId = uuidv7();
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::responses-gpt-5-mini",
      tools: [
        {
          type: "function",
          function: {
            name: "get_temperature",
            description: "Get the current temperature in a given location",
            parameters: {
              type: "object",
              properties: {
                location: {
                  type: "string",
                  description:
                    'The location to get the temperature for (e.g. "New York")',
                },
                units: {
                  type: "string",
                  description:
                    'The units to get the temperature in (must be "fahrenheit" or "celsius")',
                  enum: ["fahrenheit", "celsius"],
                },
              },
              required: ["location"],
              additionalProperties: false,
            },
          },
        },
      ],
      stream: true,
      stream_options: { include_usage: true },
      "tensorzero::episode_id": episodeId,
    });

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(0);

    // Verify consistency across chunks
    let previousInferenceId: string | null = null;
    let previousEpisodeId: string | null = null;
    let toolCallName = "";

    for (const chunk of chunks) {
      if (previousInferenceId !== null) {
        expect(chunk.id).toBe(previousInferenceId);
      }
      if (previousEpisodeId !== null) {
        // @ts-expect-error - custom TensorZero property
        expect(chunk.episode_id).toBe(previousEpisodeId);
      }
      previousInferenceId = chunk.id;
      // @ts-expect-error - custom TensorZero property
      previousEpisodeId = chunk.episode_id;

      // Check for tool call chunks and get the tool name
      if (chunk.choices && chunk.choices[0]?.delta?.tool_calls) {
        for (const toolCallDelta of chunk.choices[0].delta.tool_calls) {
          if (toolCallDelta.function?.name) {
            toolCallName += toolCallDelta.function.name;
          }
        }
      }
    }

    // Last chunk should have usage
    expect(chunks[chunks.length - 1].usage).not.toBeNull();
    expect(chunks[chunks.length - 1].usage?.prompt_tokens).toBeGreaterThan(0);
    expect(chunks[chunks.length - 1].usage?.completion_tokens).toBeGreaterThan(
      0
    );

    // Should have received a tool call for get_temperature
    expect(toolCallName).toBe("get_temperature");
  });
});
