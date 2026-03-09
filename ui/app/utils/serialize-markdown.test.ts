import { describe, it, expect } from "vitest";
import { serializeConversationMarkdown } from "./serialize-markdown";
import type { Input, StoredInference } from "~/types/tensorzero";

describe("serializeConversationMarkdown", () => {
  it("serializes a simple chat conversation", () => {
    const input: Input = {
      system: "You are a helpful assistant.",
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Hello" }],
        },
      ],
    };
    const output: StoredInference["output"] = [
      { type: "text", text: "Hi there!" },
    ];

    const result = serializeConversationMarkdown(input, output);
    expect(result).toBe(
      "## system\n\nYou are a helpful assistant.\n\n## user\n\nHello\n\n## assistant\n\nHi there!",
    );
  });

  it("handles undefined input", () => {
    const output: StoredInference["output"] = [
      { type: "text", text: "response" },
    ];
    const result = serializeConversationMarkdown(undefined, output);
    expect(result).toBe("## assistant\n\nresponse");
  });

  it("handles undefined output", () => {
    const input: Input = {
      system: "sys",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    };
    const result = serializeConversationMarkdown(input, undefined);
    expect(result).toBe("## system\n\nsys\n\n## user\n\nhi");
  });

  it("serializes JSON inference output (raw)", () => {
    const output: StoredInference["output"] = {
      raw: '{"answer": "Paris"}',
      parsed: null,
    };
    const result = serializeConversationMarkdown(undefined, output);
    expect(result).toBe("## assistant\n\n**answer**: Paris");
  });
});

describe("array indentation in markdown", () => {
  it("indents continuation lines of multi-key objects in arrays", () => {
    const output: StoredInference["output"] = {
      raw: JSON.stringify({
        people: [
          { name: "Alice", age: 30 },
          { name: "Bob", age: 25 },
        ],
      }),
      parsed: null,
    };

    const result = serializeConversationMarkdown(undefined, output);
    expect(result).toContain("- **name**: Alice\n  **age**: 30");
    expect(result).toContain("- **name**: Bob\n  **age**: 25");
  });

  it("does not indent single-line array items", () => {
    const output: StoredInference["output"] = {
      raw: JSON.stringify({ items: ["one", "two", "three"] }),
      parsed: null,
    };

    const result = serializeConversationMarkdown(undefined, output);
    expect(result).toContain("- one\n- two\n- three");
  });
});
