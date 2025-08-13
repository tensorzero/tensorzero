import type { ChatCompletionConfig } from "tensorzero-node";

import { describe, it, expect } from "vitest";
import { create_dump_variant_config } from "./variant.server";

describe("create_dump_variant_config", () => {
  it("should create correct variant config TOML with string templates", () => {
    const oldVariant: ChatCompletionConfig = {
      weight: 1,
      model: "old-model",
      json_mode: "strict" as const,
      templates: {
        system: {
          template: {
            path: "/templates/system.j2",
            contents: "This content should not appear in output",
          },
          schema: null,
        },
        user: null,
        assistant: null,
      },
      temperature: 0.5,
      top_p: 0.5,
      max_tokens: 100,
      presence_penalty: 0,
      frequency_penalty: 0,
      seed: 0,
      retries: {
        num_retries: 0,
        max_delay_s: 0,
      },
      stop_sequences: [],
    };

    const result = create_dump_variant_config(
      oldVariant,
      "new-model",
      "test_function",
    );

    expect(result).toContain("[functions.test_function.variants.new-model]");
    expect(result).toContain("weight = 0");
    expect(result).toContain('model = "new-model"');
  });

  it("should handle template objects and only serialize paths", () => {
    const oldVariant: ChatCompletionConfig = {
      weight: 1,
      model: "old-model",
      json_mode: "strict" as const,
      templates: {
        system: {
          template: {
            path: "/templates/system.j2",
            contents: "This content should not appear in output",
          },
          schema: null,
        },
        user: {
          template: {
            path: "/templates/user.j2",
            contents: "This content should not appear in output",
          },
          schema: null,
        },
        assistant: {
          template: {
            path: "/templates/assistant.j2",
            contents: "This content should not appear in output",
          },
          schema: null,
        },
      },
      temperature: 0.5,
      top_p: 0.5,
      max_tokens: 100,
      presence_penalty: 0,
      frequency_penalty: 0,
      seed: 0,
      retries: {
        num_retries: 0,
        max_delay_s: 0,
      },
      stop_sequences: [],
    };

    const result = create_dump_variant_config(
      oldVariant,
      "new-model",
      "test_function",
    );

    expect(result).toContain('system_template = "/templates/system.j2"');
    expect(result).toContain("weight = 0");
    expect(result).toContain('user_template = "/templates/user.j2"');
    expect(result).toContain('assistant_template = "/templates/assistant.j2"');
    expect(result).not.toContain("content");
  });
});
