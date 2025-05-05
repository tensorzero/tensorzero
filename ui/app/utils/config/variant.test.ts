import { type ChatCompletionConfig } from "./variant";

import { describe, it, expect } from "vitest";
import { create_dump_variant_config } from "./variant.server";

describe("create_dump_variant_config", () => {
  it("should create correct variant config TOML with string templates", () => {
    const oldVariant: ChatCompletionConfig = {
      type: "chat_completion" as const,
      weight: 1,
      model: "old-model",
      json_mode: "strict" as const,
      system_template: {
        path: "/templates/system.j2",
        content: "This content should not appear in output",
      },
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
      type: "chat_completion" as const,
      weight: 1,
      model: "old-model",
      json_mode: "strict" as const,
      system_template: {
        path: "/templates/system.j2",
        content: "This content should not appear in output",
      },
      user_template: {
        path: "/templates/user.j2",
        content: "This content should not appear in output",
      },
      assistant_template: {
        path: "/templates/assistant.j2",
        content: "This content should not appear in output",
      },
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
