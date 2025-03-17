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
import { ChatCompletionMessageParam } from "openai/resources";
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
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::basic_test",
        temperature: 0.4,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);

    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(10);
    expect(usage?.total_tokens).toBe(20);
    expect(result.choices[0].finish_reason).toBe("stop");
  });

  it("should perform basic inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::basic_test",
        temperature: 0.4,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);

    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(10);
    expect(usage?.total_tokens).toBe(20);
    expect(result.choices[0].finish_reason).toBe("stop");
  });

  it("should handle basic json schema parsing and throw proper validation error", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            name_of_assistant: "Alfred Pennyworth",
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
      client.beta.chat.completions.parse({
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
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const startTime = Date.now();

    const stream = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::basic_test",
        stream: true,
        max_tokens: 300,
        seed: 69,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

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
        expect(chunk.choices[0].delta.content).toBeNull();
        expect(chunk.usage?.prompt_tokens).toBe(10);
        expect(chunk.usage?.completion_tokens).toBe(16);
        expect(chunk.usage?.total_tokens).toBe(26);
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
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create(
        {
          messages,
          model: "tensorzero::function_name::does_not_exist",
        },
        {
          headers: {
            episode_id: episodeId,
          },
        }
      )
    ).rejects.toThrow(/Unknown function: does_not_exist/);
  });

  it("should handle streaming inference with missing function", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create(
        {
          messages,
          model: "tensorzero::function_name::",
        },
        {
          headers: {
            episode_id: episodeId,
          },
        }
      )
    ).rejects.toThrow(/function_name.*cannot be empty/);
  });

  it("should handle streaming inference with malformed function", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create(
        {
          messages,
          model: "chatgpt",
        },
        {
          headers: {
            episode_id: episodeId,
          },
        }
      )
    ).rejects.toThrow(/`model` field must start with/);
  });

  it("should handle streaming inference with missing model", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
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
            // @ts-expect-error - custom TensorZero property
            name_of_assistant: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create(
        {
          messages,
          model: "tensorzero::function_name::basic_test",
          stream: true,
        },
        {
          headers: {
            episode_id: episodeId,
          },
        }
      )
    ).rejects.toThrow(/JSON Schema validation failed/);
  });

  it("should handle tool call inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::weather_helper",
        top_p: 0.5,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::weather_helper::variant_name::variant"
    );
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls).not.toBeNull();

    const toolCalls = result.choices[0].message.tool_calls!;
    expect(toolCalls.length).toBe(1);

    const toolCall = toolCalls[0];
    expect(toolCall.type).toBe("function");
    expect(toolCall.function.name).toBe("get_temperature");
    expect(toolCall.function.arguments).toBe(
      '{"location":"Brooklyn","units":"celsius"}'
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(10);
    expect(result.choices[0].finish_reason).toBe("tool_calls");
  });

  it("should handle malformed tool call inference", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::weather_helper",
        presence_penalty: 0.5,
      },
      {
        headers: {
          episode_id: episodeId,
          variant_name: "bad_tool",
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::weather_helper::variant_name::bad_tool"
    );
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls).not.toBeNull();

    const toolCalls = result.choices[0].message.tool_calls!;
    expect(toolCalls.length).toBe(1);

    const toolCall = toolCalls[0];
    expect(toolCall.type).toBe("function");
    expect(toolCall.function.name).toBe("get_temperature");
    expect(toolCall.function.arguments).toBe(
      '{"location":"Brooklyn","units":"Celsius"}'
    );

    const usage = result.usage;
    expect(usage?.prompt_tokens).toBe(10);
    expect(usage?.completion_tokens).toBe(10);
  });

  it("should handle tool call streaming", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      {
        role: "user",
        content: "Hi I'm visiting Brooklyn from Brazil. What's the weather?",
      },
    ];

    const episodeId = uuidv7();
    const stream = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::weather_helper",
        stream: true,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

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
        expect(chunk.choices[0].delta.content).toBeNull();
        expect(chunk.choices[0].delta.tool_calls?.length).toBe(1);

        const toolCall = chunk.choices[0].delta.tool_calls![0];
        expect(toolCall.type).toBe("function");
        expect(toolCall.function?.name).toBe("get_temperature");
        expect(toolCall.function?.arguments).toBe(expectedText[i]);
      } else {
        expect(chunk.choices[0].delta.content).toBeNull();
        expect(chunk.choices[0].delta.tool_calls?.length).toBe(0);
        expect(chunk.usage?.prompt_tokens).toBe(10);
        expect(chunk.usage?.completion_tokens).toBe(5);
        expect(chunk.choices[0].finish_reason).toBe("tool_calls");
      }
    }
  });

  it("should handle JSON streaming", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            country: "Japan",
          },
        ],
      },
    ];

    const episodeId = uuidv7();
    const stream = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::json_success",
        stream: true,
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

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
        "tensorzero::function_name::json_success::variant_name::test"
      );

      if (i + 1 < chunks.length) {
        expect(chunk.choices[0].delta.content).toBe(expectedText[i]);
      } else {
        expect(chunk.choices[0].delta.content).toBe("");
        expect(chunk.usage?.prompt_tokens).toBe(10);
        expect(chunk.usage?.completion_tokens).toBe(16);
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
            "tensorzero::arguments": { assistant_name: "Alfred Pennyworth" },
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
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::json_success",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::json_success::variant_name::test"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBe('{"answer":"Hello"}');
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(10);
  });

  it("should handle json success", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            country: "Japan",
          },
        ],
      },
    ];
    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::json_success",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::json_success::variant_name::test"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBe('{"answer":"Hello"}');
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(10);
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
            // @ts-expect-error - custom TensorZero property
            country: "Japan",
          },
        ],
      },
    ];
    const episodeId = uuidv7();

    await expect(
      client.chat.completions.create(
        {
          messages,
          model: "tensorzero::function_name::json_success",
        },
        {
          headers: {
            episode_id: episodeId,
          },
        }
      )
    ).rejects.toThrow(/System message must be a text content block/);
  });

  it("should handle json failure", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Alfred Pennyworth",
          },
        ],
      },
      { role: "user", content: "Hello, world!" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::json_fail",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::json_fail::variant_name::test"
    );
    expect(result.choices[0].message.content).toBe(
      "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );
    expect(result.choices[0].message.tool_calls).toBeNull();
    expect(result.usage?.prompt_tokens).toBe(10);
    expect(result.usage?.completion_tokens).toBe(10);
  });

  it("should handle dynamic tool use inference with OpenAI", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Dr. Mehta",
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
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::basic_test",
        tools,
      },
      {
        headers: {
          episode_id: episodeId,
          variant_name: "openai",
        },
      }
    );

    expect(result.model).toBe(
      "tensorzero::function_name::basic_test::variant_name::openai"
    );
    // @ts-expect-error - custom TensorZero property
    expect(result.episode_id).toBe(episodeId);
    expect(result.choices[0].message.content).toBeNull();
    expect(result.choices[0].message.tool_calls?.length).toBe(1);

    const toolCall = result.choices[0].message.tool_calls![0];
    expect(toolCall.type).toBe("function");
    expect(toolCall.function.name).toBe("get_temperature");
    expect(JSON.parse(toolCall.function.arguments)).toEqual({
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

    const serializedOutputSchema = JSON.stringify(outputSchema);
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            assistant_name: "Dr. Mehta",
            schema: serializedOutputSchema,
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            country: "Japan",
          },
        ],
      },
    ];
    const episodeId = uuidv7();
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::dynamic_json",
        response_format: {
          type: "json_schema",
          json_schema: { name: "json_schema", ...outputSchema }, // the Node client requires a `name` field here...?
        },
      },
      {
        headers: {
          episode_id: episodeId,
          variant_name: "openai",
        },
      }
    );

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
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::model_name::openai::gpt-4o-mini",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.choices[0].message.content?.toLowerCase()).toContain("crab");
  });

  it("should handle multi-block image_base64", async () => {
    // Read image and convert to base64
    const imagePath = path.join(
      __dirname,
      "../../../tensorzero-internal/tests/e2e/providers/ferris.png"
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
    const result = await client.chat.completions.create(
      {
        messages,
        model: "tensorzero::model_name::openai::gpt-4o-mini",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    );

    expect(result.choices[0].message.content?.toLowerCase()).toContain("crab");
  });
});

it("should reject string input for function with input schema", async () => {
  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: [
        {
          // @ts-expect-error - custom TensorZero property
          assistant_name: "Alfred Pennyworth",
        },
      ],
    },
    { role: "user", content: "Hi how are you?" },
    {
      role: "user",
      content: [
        {
          // @ts-expect-error - custom TensorZero property
          country: "Japan",
        },
      ],
    },
  ];

  const episodeId = uuidv7();

  await expect(
    client.chat.completions.create(
      {
        messages,
        model: "tensorzero::function_name::json_success",
      },
      {
        headers: {
          episode_id: episodeId,
        },
      }
    )
  ).rejects.toThrow(/400 "JSON Schema validation failed fo/);
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
  const response = await client.chat.completions.create(
    {
      messages,
      model: "tensorzero::function_name::weather_helper_parallel",
      parallel_tool_calls: true,
    },
    {
      headers: {
        episode_id: episodeId,
        variant_name: "openai",
      },
    }
  );

  const assistantMessage = response.choices[0].message;
  messages.push(assistantMessage);

  expect(assistantMessage.tool_calls?.length).toBe(2);

  for (const toolCall of assistantMessage.tool_calls || []) {
    if (toolCall.function.name === "get_temperature") {
      messages.push({
        role: "tool",
        content: "70",
        tool_call_id: toolCall.id,
      });
    } else if (toolCall.function.name === "get_humidity") {
      messages.push({
        role: "tool",
        content: "30",
        tool_call_id: toolCall.id,
      });
    } else {
      throw new Error(`Unknown tool call: ${toolCall.function.name}`);
    }
  }

  const finalResponse = await client.chat.completions.create(
    {
      messages,
      model: "tensorzero::function_name::weather_helper_parallel",
    },
    {
      headers: {
        episode_id: episodeId,
        variant_name: "openai",
      },
    }
  );

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
        episode_id: episodeId,
        variant_name: "openai",
      },
      body: JSON.stringify({
        messages,
        model: "tensorzero::function_name::weather_helper_parallel",
        parallel_tool_calls: true,
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
        episode_id: episodeId,
        variant_name: "openai",
      },
      body: JSON.stringify({
        messages,
        model: "tensorzero::function_name::weather_helper_parallel",
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
