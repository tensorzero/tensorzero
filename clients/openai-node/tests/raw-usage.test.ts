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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const assertOpenAIChatRawUsageFields = (entry: any) => {
    expect(entry.data).toBeDefined();
    expect(entry.data.total_tokens).toBeDefined();
    expect(entry.data.prompt_tokens_details?.cached_tokens).toBeDefined();
    expect(
      entry.data.completion_tokens_details?.reasoning_tokens
    ).toBeDefined();
  };

  it("should return tensorzero_raw_usage in non-streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": true,
    });

    expect(result.usage).toBeDefined();
    // @ts-expect-error - custom TensorZero property at response level (sibling to usage)
    const rawUsage = result.tensorzero_raw_usage;
    expect(rawUsage).toBeDefined();
    expect(Array.isArray(rawUsage)).toBe(true);
    expect(rawUsage.length).toBeGreaterThan(0);

    // Verify structure of first entry
    const entry = rawUsage[0];
    expect(entry.model_inference_id).toBeDefined();
    expect(entry.provider_type).toBeDefined();
    expect(entry.api_type).toBeDefined();
    assertOpenAIChatRawUsageFields(entry);
  });

  it("should not return tensorzero_raw_usage when not requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": false,
    });

    expect(result.usage).toBeDefined();
    // @ts-expect-error - custom TensorZero property at response level
    const rawUsage = result.tensorzero_raw_usage;
    expect(rawUsage).toBeUndefined();
  });

  it("should return tensorzero_raw_usage in streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    // Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      stream: true,
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_usage": true,
    });

    let foundRawUsage = false;

    for await (const chunk of stream) {
      // @ts-expect-error - custom TensorZero property at chunk level (sibling to usage)
      const rawUsage = chunk.tensorzero_raw_usage;
      if (rawUsage) {
        foundRawUsage = true;
        expect(Array.isArray(rawUsage)).toBe(true);
        expect(rawUsage.length).toBeGreaterThan(0);

        const entry = rawUsage[0];
        expect(entry.model_inference_id).toBeDefined();
        expect(entry.provider_type).toBeDefined();
        expect(entry.api_type).toBeDefined();
        assertOpenAIChatRawUsageFields(entry);
      }
    }

    expect(foundRawUsage).toBe(true);
  });
});
