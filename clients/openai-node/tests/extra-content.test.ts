/**
 * Tests for `tensorzero_extra_content` round-trip support.
 *
 * These tests verify that extra content blocks (Thought, Unknown) can be:
 * 1. Received from the API in responses
 * 2. Sent back to the API in follow-up requests (round-trip)
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

interface ExtraContentBlock {
  type: "thought" | "unknown";
  insert_index?: number;
  text?: string;
  signature?: string | null;
  summary?: string | null;
  provider_type?: string | null;
  data?: unknown;
  model_name?: string | null;
  provider_name?: string | null;
}

describe("Extra Content", () => {
  it("should round-trip extra content non-streaming", async () => {
    const episodeId = uuidv7();

    // Step 1: Make inference request with a model that returns Thought content
    // The dummy::reasoner model returns [Thought, Text] content
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    const result = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::dummy::reasoner",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    // Step 2: Verify response structure
    expect(result.choices[0].message.content).toBeDefined();

    const extraContent: ExtraContentBlock[] | undefined = (
      result.choices[0].message as unknown as {
        tensorzero_extra_content?: ExtraContentBlock[];
      }
    ).tensorzero_extra_content;
    expect(extraContent).toBeDefined();
    expect(Array.isArray(extraContent)).toBe(true);
    expect(extraContent!.length).toBeGreaterThan(0);

    // Verify the structure of the thought block
    const thoughtBlock = extraContent![0];
    expect(thoughtBlock.type).toBe("thought");
    expect(thoughtBlock.insert_index).toBeDefined();
    expect(thoughtBlock.text).toBeDefined();

    // Step 3: Round-trip - send the extra content back as an assistant message
    const roundtripMessages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
      {
        role: "assistant",
        content: result.choices[0].message.content!,
        // @ts-expect-error - custom TensorZero property
        tensorzero_extra_content: extraContent,
      },
      { role: "user", content: "Continue" },
    ];

    const roundtripResult = await client.chat.completions.create({
      messages: roundtripMessages,
      model: "tensorzero::model_name::dummy::echo",
      // @ts-expect-error - custom TensorZero property
      "tensorzero::episode_id": episodeId,
    });

    // Verify round-trip succeeded
    expect(roundtripResult.choices[0].message).toBeDefined();
  });

  it("should round-trip extra content streaming", async () => {
    const episodeId = uuidv7();

    // Step 1: Make streaming inference request
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Hello" },
    ];

    // @ts-expect-error - custom TensorZero property
    const stream = await client.chat.completions.create({
      messages,
      model: "tensorzero::model_name::dummy::reasoner",
      stream: true,
      "tensorzero::episode_id": episodeId,
    });

    // Step 2: Collect chunks and extract extra content
    const extraContentChunks: ExtraContentBlock[] = [];
    let contentText = "";

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta) {
        // Collect text content
        if (delta.content) {
          contentText += delta.content;
        }

        // Collect extra content chunks
        const extraContent: ExtraContentBlock[] | undefined = (
          delta as unknown as {
            tensorzero_extra_content?: ExtraContentBlock[];
          }
        ).tensorzero_extra_content;
        if (extraContent) {
          extraContentChunks.push(...extraContent);
        }
      }
    }

    // Step 3: Verify we received extra content in streaming
    expect(extraContentChunks.length).toBeGreaterThan(0);

    // Reconstruct extra content for round-trip (filter for chunks with insert_index)
    const reconstructedExtraContent = extraContentChunks.filter(
      (chunk) => chunk.insert_index !== undefined
    );

    // Step 4: Round-trip if we have valid content
    if (reconstructedExtraContent.length > 0 && contentText.length > 0) {
      const roundtripMessages: ChatCompletionMessageParam[] = [
        { role: "user", content: "Hello" },
        {
          role: "assistant",
          content: contentText,
          // @ts-expect-error - custom TensorZero property
          tensorzero_extra_content: reconstructedExtraContent,
        },
        { role: "user", content: "Continue" },
      ];

      const roundtripResult = await client.chat.completions.create({
        messages: roundtripMessages,
        model: "tensorzero::model_name::dummy::echo",
        // @ts-expect-error - custom TensorZero property
        "tensorzero::episode_id": episodeId,
      });

      // Verify round-trip succeeded
      expect(roundtripResult.choices[0].message).toBeDefined();
    }
  });
});
