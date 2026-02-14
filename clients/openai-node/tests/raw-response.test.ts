/**
 * Tests for tensorzero::include_raw_response parameter using the OpenAI Node.js SDK.
 *
 * These tests verify that raw provider-specific response data is correctly returned
 * when tensorzero::include_raw_response is set to true via the OpenAI-compatible API.
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

describe("Raw Response", () => {
  // biome-ignore lint/suspicious/noExplicitAny: legacy test code
  const assertRawResponseEntryStructure = (entry: any) => {
    expect(entry.model_inference_id).toBeDefined();
    expect(entry.provider_type).toBeDefined();
    expect(typeof entry.provider_type).toBe("string");
    expect(entry.api_type).toBeDefined();
    expect(["chat_completions", "responses", "embeddings"]).toContain(
      entry.api_type
    );
    expect(entry.data).toBeDefined();
    expect(typeof entry.data).toBe("string");
  };

  it("should return tensorzero_raw_response in non-streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_response": true,
    });

    // @ts-expect-error - custom TensorZero property at response level
    const rawResponse = result.tensorzero_raw_response;
    expect(rawResponse).toBeDefined();
    expect(Array.isArray(rawResponse)).toBe(true);
    expect(rawResponse.length).toBeGreaterThan(0);

    // Verify structure of first entry
    const entry = rawResponse[0];
    assertRawResponseEntryStructure(entry);
  });

  it("should not return tensorzero_raw_response when not requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_response": false,
    });

    // @ts-expect-error - custom TensorZero property at response level
    const rawResponse = result.tensorzero_raw_response;
    expect(rawResponse).toBeUndefined();
  });

  it("should return tensorzero_raw_chunk in streaming response when requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      stream: true,
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_response": true,
    });

    let foundRawChunk = false;

    for await (const chunk of stream) {
      // @ts-expect-error - custom TensorZero property at chunk level
      const rawChunk = chunk.tensorzero_raw_chunk;
      if (rawChunk) {
        foundRawChunk = true;
        expect(typeof rawChunk).toBe("string");
      }

      // For single inference streaming, tensorzero_raw_response (array of previous inferences)
      // should not be present because there are no previous model inferences
    }

    expect(foundRawChunk).toBe(true);
  });

  it("should not return tensorzero_raw_chunk in streaming response when not requested", async () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const episodeId = uuidv7();
    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::gpt-4o-mini-2024-07-18",
      stream: true,
      "tensorzero::episode_id": episodeId,
      "tensorzero::include_raw_response": false,
    });

    for await (const chunk of stream) {
      // @ts-expect-error - custom TensorZero property at chunk level
      const rawChunk = chunk.tensorzero_raw_chunk;
      expect(rawChunk).toBeUndefined();
    }
  });
});
