import { describe, it, expect } from "vitest";
import {
  validateMessageLength,
  validateMessageRoles,
  validateMessage,
} from "./validation";
import type { OpenAIMessage, OpenAIRole } from "./types";

describe("convertToTiktokenModel", () => {
  it("should handle gpt-4 and gpt-4o models", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "Test" },
    ];
    expect(() =>
      validateMessageLength(messages, "gpt-4o-2024-08-06"),
    ).not.toThrow();
    expect(() =>
      validateMessageLength(messages, "gpt-4o-mini-2024-07-18"),
    ).not.toThrow();
    expect(() => validateMessageLength(messages, "gpt-4-0613")).not.toThrow();
  });

  it("should handle gpt-3.5-turbo models", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "Test" },
    ];
    expect(() =>
      validateMessageLength(messages, "gpt-3.5-turbo-0125"),
    ).not.toThrow();
    expect(() =>
      validateMessageLength(messages, "gpt-3.5-turbo-1106"),
    ).not.toThrow();
    expect(() =>
      validateMessageLength(messages, "gpt-3.5-turbo-0613"),
    ).not.toThrow();
  });

  it("should throw error for unsupported models", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "Test" },
    ];
    expect(() => validateMessageLength(messages, "unsupported-model")).toThrow(
      "Unsupported model: unsupported-model. Supported models are: gpt-4o-*, gpt-4-*, gpt-3.5-turbo*",
    );
  });
});

describe("validateMessageLength", () => {
  it("should validate messages within token limit", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    const result = validateMessageLength(messages, "gpt-3.5-turbo");
    expect(result.isValid).toBe(true);
    expect(result.tokenCount).toBeLessThan(4096);
  });

  it("should handle messages with undefined content", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      {
        role: "assistant" as OpenAIRole,
        tool_calls: [
          {
            id: "1",
            type: "function",
            function: { name: "test", arguments: "{}" },
          },
        ],
      },
    ];

    const result = validateMessageLength(messages, "gpt-3.5-turbo");
    expect(result.isValid).toBe(true);
  });

  it("should validate custom token limit", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
    ];

    const result = validateMessageLength(messages, "gpt-3.5-turbo", 10);
    expect(result.isValid).toBe(false);
  });
});

describe("validateMessageRoles", () => {
  it("should validate messages with all required roles", () => {
    const messages: OpenAIMessage[] = [
      { role: "user" as OpenAIRole, content: "Hello!" },
      { role: "assistant" as OpenAIRole, content: "Hi there!" },
    ];

    const result = validateMessageRoles(messages);
    expect(result.isValid).toBe(true);
    expect(result.missingRoles).toHaveLength(0);
  });

  it("should validate messages with optional roles", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
      { role: "assistant" as OpenAIRole, content: "Hi there!" },
      {
        role: "tool" as OpenAIRole,
        content: "Some tool response",
        tool_call_id: "1",
      },
    ];

    const result = validateMessageRoles(messages);
    expect(result.isValid).toBe(true);
    expect(result.missingRoles).toHaveLength(0);
  });

  it("should detect missing roles", () => {
    const testCases = [
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
          { role: "assistant" as OpenAIRole, content: "Hi there!" },
        ],
        missing: ["user"],
      },
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
          { role: "user" as OpenAIRole, content: "Hello!" },
        ],
        missing: ["assistant"],
      },
      {
        messages: [
          {
            role: "tool" as OpenAIRole,
            content: "Some tool response",
            tool_call_id: "1",
          },
        ],
        missing: ["user", "assistant"],
      },
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
        ],
        missing: ["user", "assistant"],
      },
    ];

    for (const testCase of testCases) {
      const result = validateMessageRoles(testCase.messages);
      expect(result.isValid).toBe(false);
      for (const role of testCase.missing) {
        expect(result.missingRoles).toContain(role);
      }
    }
  });
});

describe("validateMessage", () => {
  it("should validate valid messages", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    const result = validateMessage(messages, "gpt-3.5-turbo");
    expect(result.isValid).toBe(true);
    expect(result.lengthValidation.isValid).toBe(true);
    expect(result.rolesValidation.isValid).toBe(true);
  });

  it("should validate messages with custom token limit", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    const result = validateMessage(messages, "gpt-3.5-turbo", 10);
    expect(result.isValid).toBe(false);
    expect(result.lengthValidation.isValid).toBe(false);
    expect(result.rolesValidation.isValid).toBe(true);
  });

  it("should validate messages with missing roles", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    const result = validateMessage(messages, "gpt-3.5-turbo");
    expect(result.isValid).toBe(false);
    expect(result.lengthValidation.isValid).toBe(true);
    expect(result.rolesValidation.isValid).toBe(false);
    expect(result.rolesValidation.missingRoles).toContain("user");
  });
});
