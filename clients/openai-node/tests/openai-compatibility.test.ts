/**
 * Tests for the TensorZero OpenAI-compatible endpoint using the OpenAI Node.js client
 *
 * These tests cover the major functionality of the translation
 * layer between the OpenAI interface and TensorZero. They do not
 * attempt to comprehensively cover all of TensorZero's functionality.
 * See the tests across the Rust codebase for more comprehensive tests.
 */
import { describe, it, expect, beforeAll } from "vitest";
import OpenAI from "openai";
import {
  ChatCompletionMessageParam,
  ChatCompletionMessageFunctionToolCall,
} from "openai/resources";
import { v7 as uuidv7 } from "uuid";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Client setup
let client: OpenAI;

beforeAll(() => {
  client = new OpenAI({
    apiKey: "donotuse",
    baseURL: "http://127.0.0.1:3000/openai/v1",
  });
});

describe("OpenAI Compatibility", () => {
  it("should perform basic inference with old model format", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      temperature: 0.4,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);

    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(1);
    expect(usage?.total_tokens).toBe(11);
    expect(result.choices[0].finish_reason).toBe("stop");
  });

  it("should perform basic inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      temperature: 0.4,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::tags": {
        foo: "bar",
      },
    });

    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);

    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(1);
    expect(usage?.total_tokens).toBe(11);
    expect(result.choices[0].finish_reason).toBe("stop");
  });

  it("should handle basic json schema parsing and throw proper validation error", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              name_of_assistant: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    // Define a schema for response parsing - should fail in this case
    const responseSchema = {
      name: { type: "string" },
    };

    await expect(
      client.chat.completions.parse({
        messages,
        model: "tensorzero::function_name::basic_test",
        temperature: 0.4,
        response_schema: responseSchema,
        episode_id: episodeId,
      })
    ).rejects.toThrow();
  });

  it("should handle streaming inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const startTime = Date.now();

    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      stream: true,
      max_tokens: 300,
      seed: 69,
      "tensorzero::episode_id": episodeId,
    });

    let firstChunkDuration: number | null = null;
    const chunks = [];

    for await (const chunk of stream) {
      chunks.push(chunk);
      if (firstChunkDuration === null) {
        firstChunkDuration = Date.now() - startTime;
      }
    }

    const lastChunkDuration =
      Date.now() - startTime - (firstChunkDuration || 0);
    expect(lastChunkDuration).toBeGreaterThan(firstChunkDuration! + 100);

    const expectedText = [
      "Wally,",
      " the",
      " golden",
      " retriever,",
      " wagged",
      " his",
      " tail",
      " excitedly",
      " as",
      " he",
      " devoured",
      " a",
      " slice",
      " of",
      " cheese",
      " pizza.",
    ];

    let previousInferenceId: string | null = null;
    let previousEpisodeId: string | null = null;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
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

      expect(chunk.model).toBe(
        "tensorzero::function_name::basic_test::variant_name::test"
      );

      if (i + 1 < chunks.length) {
        expect(chunk.choices.length).toBe(1);
        expect(chunk.choices[0].delta.content).toBe(expectedText[i]);
        expect(chunk.choices[0].finish_reason).toBeNull();
      } else {
        expect(chunk.choices[0].delta.content).toBeUndefined();
        // No usage information because we didn't set `include_usage`
        expect(chunk.usage).toBeNull();
        expect(chunk.choices[0].finish_reason).toBe("stop");
      }
    }
  });

  it("should handle streaming inference with nonexistent function", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create({
        messages,
        model: "tensorzero::function_name::does_not_exist",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      })
    ).rejects.toThrow(/Unknown function: does_not_exist/);
  });

  it("should handle streaming inference with missing function", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create({
        messages,
        model: "tensorzero::function_name::",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      })
    ).rejects.toThrow(/function_name.*cannot be empty/);
  });

  it("should handle streaming inference with malformed function", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create({
        messages,
        model: "chatgpt",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      })
    ).rejects.toThrow(/`model` field must start with/);
  });

  it("should handle streaming inference with missing model", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    await expect(
      // @ts-expect-error - missing model
      client.chat.completions.create({
        messages,
      })
    ).rejects.toThrow(/missing field `model`/);
  });

  it("should handle streaming inference with malformed input", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              name_of_assistant: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      // @ts-expect-error - custom TensorZero property
      client.chat.completions.create({
        messages,
        model: "tensorzero::function_name::basic_test",
        stream: true,
        "tensorzero::episode_id": episodeId,
      })
    ).rejects.toThrow(/JSON Schema validation failed/);
  });

  it("should handle tool call inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::weather_helper",
      top_p: 0.5,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.model).toBe(
      "tensorzero::function_name::weather_helper::variant_name::variant"
    );
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls).not.toBeNull();

    const toolCalls = result.choices[0].message.tool_calls!;
    expect(toolCalls.length).toBe(1);

    const toolCall = toolCalls[0];
    expect(toolCall.type).toBe("function");
    expect(
      (toolCall as ChatCompletionMessageFunctionToolCall).function.name
    ).toBe("get_temperature");
    expect(
      (toolCall as ChatCompletionMessageFunctionToolCall).function.arguments
    ).toBe('{"location":"Brooklyn","units":"celsius"}');

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(1);
    expect(result.choices[0].finish_reason).toBe("tool_calls");
  });

  it("should handle malformed tool call inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::weather_helper",
      presence_penalty: 0.5,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::variant_name": "bad_tool",
    });

    expect(result.model).toBe(
      "tensorzero::function_name::weather_helper::variant_name::bad_tool"
    );
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls).not.toBeNull();

    const toolCalls = result.choices[0].message.tool_calls!;
    expect(toolCalls.length).toBe(1);

    const toolCall = toolCalls[0];
    expect(toolCall.type).toBe("function");
    expect(
      (toolCall as ChatCompletionMessageFunctionToolCall).function.name
    ).toBe("get_temperature");
    expect(
      (toolCall as ChatCompletionMessageFunctionToolCall).function.arguments
    ).toBe('{"location":"Brooklyn","units":"Celsius"}');

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(1);
  });

  it("should handle tool call streaming", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    // @ts-expect-error - custom TensorZero property

    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::weather_helper",
      stream: true,
      "tensorzero::episode_id": episodeId,
    });

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const expectedText = [
      '{"location"',
      ':"Brooklyn"',
      ',"units"',
      ':"celsius',
      '"}',
    ];

    let previousInferenceId: string | null = null;
    let previousEpisodeId: string | null = null;
    let nameSeen = false;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
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

      expect(chunk.model).toBe(
        "tensorzero::function_name::weather_helper::variant_name::variant"
      );

      if (i + 1 < chunks.length) {
        expect(chunk.choices.length).toBe(1);
        expect(chunk.choices[0].delta.content).toBeUndefined();
        expect(chunk.choices[0].delta.tool_calls?.length).toBe(1);

        const toolCall = chunk.choices[0].delta.tool_calls![0];
        expect(toolCall.type).toBe("function");
        if (toolCall.function?.name) {
          expect(nameSeen).toBe(false);
          expect(toolCall.function.name).toBe("get_temperature");
          nameSeen = true;
        }
        expect(toolCall.function?.arguments).toBe(expectedText[i]);
      } else {
        expect(chunk.choices[0].delta.content).toBeUndefined();
        expect(chunk.choices[0].delta.tool_calls).toBeUndefined();
        // No usage information because we didn't set `include_usage`
        expect(chunk.usage).toBeNull();
        expect(chunk.choices[0].finish_reason).toBe("tool_calls");
      }
    }
    expect(nameSeen).toBe(true);
  });

  it("should handle JSON streaming", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              country: "Japan",
            },
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::json_success",
      stream: true,
      "tensorzero::episode_id": episodeId,
      "tensorzero::variant_name": "test-diff-schema",
    });

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const expectedText = [
      "Wally,",
      " the",
      " golden",
      " retriever,",
      " wagged",
      " his",
      " tail",
      " excitedly",
      " as",
      " he",
      " devoured",
      " a",
      " slice",
      " of",
      " cheese",
      " pizza.",
    ];

    let previousInferenceId: string | null = null;
    let previousEpisodeId: string | null = null;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
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

      expect(chunk.model).toBe(
        "tensorzero::function_name::json_success::variant_name::test-diff-schema"
      );

      if (i + 1 < chunks.length) {
        expect(chunk.choices[0].delta.content).toBe(expectedText[i]);
      } else {
        expect(chunk.choices[0].delta.content).toBe("");
        // No usage information because we didn't set `include_usage`
        expect(chunk.usage).toBeNull();
      }
    }
  });

  it("should handle json success with non-deprecated format", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": { country: "Japan" },
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::json_success",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.model).toBe(
      "tensorzero::function_name::json_success::variant_name::test"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBe('{"answer":"Hello"}');
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(1);
  });

  it("should handle json success", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              country: "Japan",
            },
          },
        ],
      },
    ];
    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::json_success",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.model).toBe(
      "tensorzero::function_name::json_success::variant_name::test"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBe('{"answer":"Hello"}');
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(1);
  });

  it("should handle json invalid system", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "image_url",
            // @ts-expect-error - invalid system message
            image_url: { url: "https://example.com/image.jpg" },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              country: "Japan",
            },
          },
        ],
      },
    ];
    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create({
        messages,
        model: "tensorzero::function_name::json_success",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      })
    ).rejects.toThrow(
      /System message must contain only text or template content blocks/
    );
  });

  it("should handle json failure", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello, world!" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::json_fail",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.model).toBe(
      "tensorzero::function_name::json_fail::variant_name::test"
    );
    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(1);
  });

  it("should handle caching", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      temperature: 0.4,
    });

    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(1);
    expect(usage?.total_tokens).toBe(11);
    // Sleep so we're sure the cache is updated
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Test caching
    const cachedResult = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      temperature: 0.4,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::cache_options": {
        max_age_s: 10,
        enabled: "on",
      },
    });

    expect(cachedResult.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const cachedUsage = cachedResult.usage;
    expect(cachedUsage?.prompt_tokens).toBe(0); // Should be cached
    expect(cachedUsage?.completion_tokens).toBe(0); // Should be cached
    expect(cachedUsage?.total_tokens).toBe(0); // Should be cached
  });

  it("should handle streaming caching", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    // First streaming request
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      stream: true,
      stream_options: {
        include_usage: true,
      },
      seed: 69,
    });

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const expectedText = [
      "Wally,",
      " the",
      " golden",
      " retriever,",
      " wagged",
      " his",
      " tail",
      " excitedly",
      " as",
      " he",
      " devoured",
      " a",
      " slice",
      " of",
      " cheese",
      " pizza.",
    ];

    let content = "";
    for (let i = 0; i < chunks.length - 1; i++) {
      if (i < expectedText.length) {
        expect(chunks[i].choices[0].delta.content).toBe(expectedText[i]);
        content += chunks[i].choices[0].delta.content;
      }
    }

    const prevChunk = chunks[chunks.length - 2];
    expect(prevChunk.choices[0].finish_reason).toBe("stop");
    expect(prevChunk.usage).toBeNull();

    // Check final chunk (which contains usage)
    const finalChunk = chunks[chunks.length - 1];
    expect(finalChunk.usage?.prompt_tokens).toBe(10);
    expect(finalChunk.usage?.completion_tokens).toBe(16);
    expect(finalChunk.choices).toStrictEqual([]);

    // Sleep so we're sure the cache is warmed up
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Second streaming request with cache
    // @ts-expect-error - custom TensorZero property
    const cachedStream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      stream: true,
      stream_options: {
        include_usage: true,
      },
      seed: 69,
      "tensorzero::cache_options": {
        max_age_s: 30, // NOTE: This was 10s and we actually occasionally time out on CI.
        enabled: "on",
      },
    });

    const cachedChunks = [];
    for await (const chunk of cachedStream) {
      cachedChunks.push(chunk);
    }

    let cachedContent = "";
    for (let i = 0; i < cachedChunks.length - 1; i++) {
      if (i < expectedText.length) {
        expect(cachedChunks[i].choices[0].delta.content).toBe(expectedText[i]);
        cachedContent += cachedChunks[i].choices[0].delta.content;
      }
    }

    expect(content).toBe(cachedContent);

    const prevCachedChunk = cachedChunks[cachedChunks.length - 2];
    expect(prevCachedChunk.choices[0].finish_reason).toBe("stop");
    expect(prevCachedChunk.usage).toBeNull();

    // Check final cached chunk
    const finalCachedChunk = cachedChunks[cachedChunks.length - 1];
    expect(finalCachedChunk.usage?.prompt_tokens).toBe(0);
    expect(finalCachedChunk.usage?.completion_tokens).toBe(0);
    expect(finalCachedChunk.usage?.total_tokens).toBe(0);
  });

  it("should handle dynamic tool use inference with OpenAI", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Dr. Mehta",
            },
          },
        ],
      },
      {
        role: "user",
        content:
          "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function.",
      },
    ];

    const tools = [
      {
        type: "function" as const,
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
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      tools,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::variant_name": "openai",
    });

    expect(result.model).toBe(
      "tensorzero::function_name::basic_test::variant_name::openai"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls?.length).toBe(1);

    const toolCall = result.choices[0].message.tool_calls![0];
    expect(toolCall.type).toBe("function");
    expect(
      (toolCall as ChatCompletionMessageFunctionToolCall).function.name
    ).toBe("get_temperature");
    expect(
      JSON.parse(
        (toolCall as ChatCompletionMessageFunctionToolCall).function.arguments
      )
    ).toEqual({
      location: "Tokyo",
      units: "celsius",
    });

    expect(result.usage?.prompt_tokens).toBeGreaterThan(100);
    expect(result.usage?.completion_tokens).toBeGreaterThan(10);
  });

  it("should handle dynamic json mode inference with OpenAI", async () => {
    const outputSchema = {
      type: "object",
      properties: { response: { type: "string" } },
      required: ["response"],
      additionalProperties: false,
    };
    const response_format = {
      type: "json_schema" as const,
      json_schema: {
        name: "test",
        description: "test",
        schema: outputSchema,
        strict: true,
      },
    };

    const serializedOutputSchema = JSON.stringify(outputSchema);
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              assistant_name: "Dr. Mehta",
              schema: serializedOutputSchema,
            },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            // @ts-expect-error - custom TensorZero property
            "tensorzero::arguments": {
              country: "Japan",
            },
          },
        ],
      },
    ];
    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::dynamic_json",
      response_format,
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::variant_name": "openai",
    });

    expect(result.model).toBe(
      "tensorzero::function_name::dynamic_json::variant_name::openai"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);

    const jsonContent = JSON.parse(result.choices[0].message.content!);
    expect(jsonContent.response.toLowerCase()).toContain("tokyo");
    expect(result.choices[0].message.tool_calls).toBeNull();

    expect(result.usage?.prompt_tokens).toBeGreaterThan(50);
    expect(result.usage?.completion_tokens).toBeGreaterThan(0);
  });

  it("should handle multi-block image_url", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Output exactly two words describing the image",
          },
          {
            type: "image_url",
            image_url: {
              url: "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
            },
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::openai::gpt-4o-mini",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.choices[0].message.content?.toLowerCase()).toContain("crab");
  });

  it("should handle multi-block image_base64", async () => {
    // Read image and convert to base64
    const imagePath = path.join(
      __dirname,
      "../../../tensorzero-core/tests/e2e/providers/ferris.png"
    );
    const ferrisPng = fs.readFileSync(imagePath).toString("base64");

    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Output exactly two words describing the image",
          },
          {
            type: "image_url",
            image_url: {
              url: `data:image/png;base64,${ferrisPng}`,
            },
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::openai::gpt-4o-mini",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.choices[0].message.content?.toLowerCase()).toContain("crab");
  });

  it("should handle multi-block file_base64", async () => {
    // Read PDF file and convert to base64
    const pdfPath = path.join(
      __dirname,
      "../../../tensorzero-core/tests/e2e/providers/deepseek_paper.pdf"
    );
    const deepseekPaperPdf = fs.readFileSync(pdfPath).toString("base64");

    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Output exactly two words describing the image",
          },
          {
            type: "file",
            file: {
              file_data: deepseekPaperPdf,
              filename: "test.pdf",
            },
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::dummy::require_pdf",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    expect(result.choices[0].message.content).not.toBeNull();
    const jsonContent = JSON.parse(result.choices[0].message.content!);
    expect(jsonContent[0].Base64.storage_path).toEqual({
      kind: { type: "disabled" },
      path: "observability/files/3e127d9a726f6be0fd81d73ccea97d96ec99419f59650e01d49183cd3be999ef.pdf",
    });
  });
});

it("should reject string input for function with input schema", async () => {
  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: [
        {
          type: "text",
          // @ts-expect-error - custom TensorZero property
          "tensorzero::arguments": {
            assistant_name: "Alfred Pennyworth",
          },
        },
      ],
    },
    { role: "user", content: "Hi how are you?" },
    {
      role: "user",
      content: [
        {
          type: "text",
          // @ts-expect-error - custom TensorZero property
          "tensorzero::arguments": {
            country: "Japan",
          },
        },
      ],
    },
  ];

  const episodeId = uuidv7();

  await expect(
    client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::json_success",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    })
  ).rejects.toThrow(/400 "JSON Schema validation failed/);
});

it("should handle multi-turn parallel tool calls", async () => {
  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: [
        {
          type: "text",
          // @ts-expect-error - custom TensorZero property
          "tensorzero::arguments": { assistant_name: "Dr. Mehta" },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "What is the weather like in Tokyo? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.",
        },
      ],
    },
  ];

  const episodeId = uuidv7();
  const response = await client.chat.completions.create({
    messages,
    model: "tensorzero::function_name::weather_helper_parallel",
    parallel_tool_calls: true,
    // @ts-expect-error - custom TensorZero property
    "tensorzero::episode_id": episodeId,
    "tensorzero::variant_name": "openai",
  });

  const assistantMessage = response.choices[0].message;
  messages.push(assistantMessage);

  expect(assistantMessage.tool_calls?.length).toBe(2);

  for (const toolCall of assistantMessage.tool_calls || []) {
    const functionToolCall = toolCall as ChatCompletionMessageFunctionToolCall;
    if (functionToolCall.function.name === "get_temperature") {
      messages.push({
        role: "tool",
        content: "70",
        tool_call_id: toolCall.id,
      });
    } else if (functionToolCall.function.name === "get_humidity") {
      messages.push({
        role: "tool",
        content: "30",
        tool_call_id: toolCall.id,
      });
    } else {
      throw new Error(`Unknown tool call: ${functionToolCall.function.name}`);
    }
  }

  const finalResponse = await client.chat.completions.create({
    messages,
    model: "tensorzero::function_name::weather_helper_parallel",
    // @ts-expect-error - custom TensorZero property
    "tensorzero::episode_id": episodeId,
    "tensorzero::variant_name": "openai",
  });

  const finalAssistantMessage = finalResponse.choices[0].message;

  expect(finalAssistantMessage.content).toContain("70");
  expect(finalAssistantMessage.content).toContain("30");
});

it("should handle multi-turn parallel tool calls using TensorZero gateway directly", async () => {
  // Define types similar to OpenAI's ChatCompletionMessageParam but simpler for direct API use
  type MessageContent = {
    type: string;
    text?: string;
    "tensorzero::arguments"?: Record<string, string>;
  };

  type Message = {
    role: string;
    content: MessageContent[];
    tool_call_id?: string;
  };

  const messages: Message[] = [
    {
      role: "system",
      content: [
        {
          type: "text",
          "tensorzero::arguments": { assistant_name: "Dr. Mehta" },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.",
        },
      ],
    },
  ];

  const episodeId = uuidv7();

  // First request to get tool calls
  const firstResponse = await fetch(
    "http://127.0.0.1:3000/openai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer donotuse",
      },
      body: JSON.stringify({
        messages,
        model: "tensorzero::function_name::weather_helper_parallel",
        parallel_tool_calls: true,
        "tensorzero::episode_id": episodeId,
        "tensorzero::variant_name": "openai",
      }),
    }
  );

  if (!firstResponse.ok) {
    throw new Error(`HTTP error! status: ${firstResponse.status}`);
  }

  const firstResponseData = await firstResponse.json();
  const assistantMessage = firstResponseData.choices[0].message;
  messages.push(assistantMessage as Message);

  expect(assistantMessage.tool_calls.length).toBe(2);

  // Add tool responses
  for (const toolCall of assistantMessage.tool_calls) {
    if (toolCall.function.name === "get_temperature") {
      messages.push({
        role: "tool",
        content: [{ type: "text", text: "70" }],
        tool_call_id: toolCall.id,
      });
    } else if (toolCall.function.name === "get_humidity") {
      messages.push({
        role: "tool",
        content: [{ type: "text", text: "30" }],
        tool_call_id: toolCall.id,
      });
    } else {
      throw new Error(`Unknown tool call: ${toolCall.function.name}`);
    }
  }

  // Second request with tool responses
  const secondResponse = await fetch(
    "http://127.0.0.1:3000/openai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer donotuse",
      },
      body: JSON.stringify({
        messages,
        model: "tensorzero::function_name::weather_helper_parallel",
        "tensorzero::episode_id": episodeId,
        "tensorzero::variant_name": "openai",
      }),
    }
  );

  if (!secondResponse.ok) {
    throw new Error(`HTTP error! status: ${secondResponse.status}`);
  }

  const secondResponseData = await secondResponse.json();
  const finalAssistantMessage = secondResponseData.choices[0].message;

  expect(finalAssistantMessage.content).toContain("70");
  expect(finalAssistantMessage.content).toContain("30");
});

it("should handle chat function null response", async () => {
  const result = await client.chat.completions.create({
    model: "tensorzero::function_name::null_chat",
    messages: [
      {
        role: "user",
        content: "No yapping!",
      },
    ],
  });

  expect(result.model).toBe(
    "tensorzero::function_name::null_chat::variant_name::variant"
  );
  expect(result.choices[0].message.content).toBeNull();
});

it("should handle json function null response", async () => {
  const result = await client.chat.completions.create({
    model: "tensorzero::function_name::null_json",
    messages: [
      {
        role: "user",
        content: "Extract no data!",
      },
    ],
  });

  expect(result.model).toBe(
    "tensorzero::function_name::null_json::variant_name::variant"
  );
  expect(result.choices[0].message.content).toBeNull();
});

it("should handle extra headers parameter", async () => {
  const result = await client.chat.completions.create({
    // @ts-expect-error - custom TensorZero property
    "tensorzero::extra_headers": [
      {
        model_provider_name:
          "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
        name: "x-my-extra-header",
        value: "my-extra-header-value",
      },
    ],
    messages: [{ role: "user", content: "Hello, world!" }],
    model: "tensorzero::model_name::dummy::echo_extra_info",
  });

  expect(result.model).toBe("tensorzero::model_name::dummy::echo_extra_info");
  expect(JSON.parse(result.choices[0].message.content!)).toEqual({
    extra_body: { inference_extra_body: [] },
    extra_headers: {
      inference_extra_headers: [
        {
          model_provider_name:
            "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
          name: "x-my-extra-header",
          value: "my-extra-header-value",
        },
      ],
      variant_extra_headers: null,
    },
  });
});

it("should handle extra body parameter", async () => {
  const result = await client.chat.completions.create({
    // @ts-expect-error - custom TensorZero property
    "tensorzero::extra_body": [
      {
        model_provider_name:
          "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
        pointer: "/thinking",
        value: {
          type: "enabled",
          budget_tokens: 1024,
        },
      },
    ],
    messages: [{ role: "user", content: "Hello, world!" }],
    model: "tensorzero::model_name::dummy::echo_extra_info",
  });

  expect(result.model).toBe("tensorzero::model_name::dummy::echo_extra_info");
  expect(JSON.parse(result.choices[0].message.content!)).toEqual({
    extra_body: {
      inference_extra_body: [
        {
          model_provider_name:
            "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
          pointer: "/thinking",
          value: { type: "enabled", budget_tokens: 1024 },
        },
      ],
    },
    extra_headers: { variant_extra_headers: null, inference_extra_headers: [] },
  });
});

it("should handle multiple text blocks in message", async () => {
  const result = await client.chat.completions.create({
    model: "tensorzero::model_name::dummy::multiple-text-blocks",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Hello, world!" },
          { type: "text", text: "Hello, world!" },
        ],
      },
    ],
  });

  expect(result.model).toBe(
    "tensorzero::model_name::dummy::multiple-text-blocks"
  );
  expect(result.choices[0].message.content).toContain("Megumin");
});
