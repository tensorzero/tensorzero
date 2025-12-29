/**
 * Tests for tensorzero::include_raw_usage parameter using the OpenAI Node.js SDK.
 *
 * These tests verify that raw provider-specific usage data is correctly returned
 * when tensorzero::include_raw_usage is set to true via the OpenAI-compatible API.
 */
import { describe, it, expect, beforeAll } from "vitest";
import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources";
import { v7 as uuidv7 } from "uuid";

let client: OpenAI;

beforeAll(() => {
  client = new OpenAI({
    apiKey: "donotuse",
    baseURL: "http://127.0.0.1:3000/openai/v1",
  });
});

describe("Raw Usage", () => {
  it("should return tensorzero_raw_usage in non-streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            type: "tensorzero::template",
            name: "system",
            arguments: {
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
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": true,
    });

    expect(result.usage).toBeDefined();
    // @ts-expect-error - custom TensorZero property
    const rawUsage = result.usage?.tensorzero_raw_usage;
    expect(rawUsage).toBeDefined();
    expect(Array.isArray(rawUsage)).toBe(true);
    expect(rawUsage.length).toBeGreaterThan(0);

    // Verify structure of first entry
    const entry = rawUsage[0];
    expect(entry.model_inference_id).toBeDefined();
    expect(entry.provider_type).toBeDefined();
    expect(entry.api_type).toBeDefined();
  });

  it("should not return tensorzero_raw_usage when not requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            type: "tensorzero::template",
            name: "system",
            arguments: {
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
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": false,
    });

    expect(result.usage).toBeDefined();
    // @ts-expect-error - custom TensorZero property
    const rawUsage = result.usage?.tensorzero_raw_usage;
    expect(rawUsage).toBeUndefined();
  });

  it("should return tensorzero_raw_usage in streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          {
            // @ts-expect-error - custom TensorZero property
            type: "tensorzero::template",
            name: "system",
            arguments: {
              assistant_name: "Alfred Pennyworth",
            },
          },
        ],
      },
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    // Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::function_name::basic_test",
      stream: true,
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": true,
    });

    let foundRawUsage = false;

    for await (const chunk of stream) {
      if (chunk.usage) {
        // @ts-expect-error - custom TensorZero property
        const rawUsage = chunk.usage?.tensorzero_raw_usage;
        if (rawUsage) {
          foundRawUsage = true;
          expect(Array.isArray(rawUsage)).toBe(true);
          expect(rawUsage.length).toBeGreaterThan(0);

          const entry = rawUsage[0];
          expect(entry.model_inference_id).toBeDefined();
          expect(entry.provider_type).toBeDefined();
          expect(entry.api_type).toBeDefined();
        }
      }
    }

    expect(foundRawUsage).toBe(true);
  });
});
