/**
 * Tests for OTLP traces extra headers using the OpenAI Node.js SDK
 *
 * These tests verify that custom OTLP headers can be sent via the OpenAI SDK's
 * headers option (in the second parameter) to the TensorZero OpenAI-compatible
 * endpoint and are correctly exported to Tempo.
 */
import { describe, it, expect, beforeAll } from "vitest";
import OpenAI from "openai";
import { v7 as uuidv7 } from "uuid";

// Client setup
let client: OpenAI;

beforeAll(() => {
  client = new OpenAI({
    apiKey: "not-used",
    baseURL: "http://127.0.0.1:3000/openai/v1",
  });
});

describe("OTLP Traces Extra Headers", () => {
  it("should handle OTLP traces extra headers with Tempo", async () => {
    // Use a unique header value to identify this specific trace
    const testValue = `openai-node-test-${uuidv7()}`;

    const result = await client.chat.completions.create(
      {
        model: "tensorzero::function_name::basic_test",
        messages: [
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
          { role: "user", content: "What is 4+4?" },
        ],
        "tensorzero::variant_name": "openai",
      },
      {
        headers: {
          "tensorzero-otlp-traces-extra-header-x-dummy-tensorzero": testValue,
        },
      }
    );

    const inferenceId = result.id;

    // Wait for trace to be exported to Tempo (same as other Tempo tests)
    await new Promise((resolve) => setTimeout(resolve, 25000));

    // Query Tempo for the trace
    const tempoUrl =
      process.env.TENSORZERO_TEMPO_URL || "http://localhost:3200";
    const startTime = Math.floor(Date.now() / 1000) - 60; // Look back 60 seconds
    const endTime = Math.floor(Date.now() / 1000);

    const searchUrl = `${tempoUrl}/api/search?tags=inference_id=${inferenceId}&start=${startTime}&end=${endTime}`;
    const searchResponse = await fetch(searchUrl);
    expect(searchResponse.status).toBe(200);

    const tempoTraces = await searchResponse.json();
    expect(tempoTraces.traces.length).toBeGreaterThan(0);

    const traceId = tempoTraces.traces[0].traceID;

    // Get trace details
    const traceUrl = `${tempoUrl}/api/traces/${traceId}`;
    const traceResponse = await fetch(traceUrl);
    expect(traceResponse.status).toBe(200);

    const traceData = await traceResponse.json();

    // Find the parent span (POST /openai/v1/chat/completions) and check for our custom header
    let foundHeader = false;
    for (const batch of traceData.batches || []) {
      for (const scopeSpan of batch.scopeSpans || []) {
        for (const span of scopeSpan.spans || []) {
          if (span.name === "POST /openai/v1/chat/completions") {
            // Check span attributes for our custom header value
            for (const attr of span.attributes || []) {
              if (attr.key === "tensorzero.custom_key") {
                const attrValue = attr.value?.stringValue;
                if (attrValue === testValue) {
                  foundHeader = true;
                  break;
                }
              }
            }
          }
        }
      }
    }

    expect(foundHeader).toBe(true);
  }, 35000); // Set timeout to 35 seconds to account for 25s wait
});
