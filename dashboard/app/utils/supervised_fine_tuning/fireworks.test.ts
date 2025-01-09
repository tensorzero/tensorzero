import { describe, it, expect } from "vitest";
import { tensorzero_inference_to_fireworks_messages } from "./fireworks";
import { create_env } from "../minijinja/pkg/minijinja_bindings";
import type { ParsedInferenceExample } from "../clickhouse/curation";

describe("tensorzero_inference_to_fireworks_messages", async () => {
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
    } as ParsedInferenceExample;

    const fireworksExample = tensorzero_inference_to_fireworks_messages(
      row,
      env,
    );
    expect(fireworksExample.messages.length).toBe(3);
    expect(fireworksExample.messages[0]).toStrictEqual({
      role: "system",
      content: "Do NER Properly!",
    });
    expect(fireworksExample.messages[1]).toStrictEqual({
      role: "user",
      content:
        'We have in no way seen any Iraqi troops in the city or in its approaches , " a U.N. relief official told Reuters .',
    });
    expect(fireworksExample.messages[2]).toStrictEqual({
      role: "assistant",
      content:
        '{"person":[],"organization":["U.N.","Reuters"],"location":[],"miscellaneous":["Iraqi"]}',
    });
  });
  it("test chat", () => {
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
        ],
      },
      output: [{ type: "text", text: "it is 34 and sunny" }],
      episode_id: "0192ced0-a2c6-7323-be23-ce4124e683d3",
    } as ParsedInferenceExample;

    const fireworksExample = tensorzero_inference_to_fireworks_messages(
      row,
      env,
    );
    expect(fireworksExample.messages.length).toBe(3);
    expect(fireworksExample.messages[0]).toStrictEqual({
      role: "system",
      content: "Help me out with the weather!",
    });
    expect(fireworksExample.messages[1]).toStrictEqual({
      role: "user",
      content: "What is the weather in Columbus?",
    });
    expect(fireworksExample.messages[2]).toStrictEqual({
      role: "assistant",
      content: "it is 34 and sunny",
    });
  });
});
