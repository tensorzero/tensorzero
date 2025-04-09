import { describe, it, expect } from "vitest";
import {
  validateMessageLength,
  validateMessageRoles,
  validateMessage,
  validateDataFormat,
  validateDataset,
  calculateDistribution,
  analyzeDataset,
} from "./validation";
import { MODEL_TOKEN_LIMITS } from "./constants";
import {
  getModelTokenLimit,
  countAssistantTokens,
  getEncodingForModel,
} from "./openAITokenCounter";
import type { OpenAIMessage, OpenAIRole } from "./types";
import { CURRENT_MODEL_VERSIONS } from "./constants";

describe("getModelTokenLimit", () => {
  it("should return correct token limits for specific models", () => {
    expect(getModelTokenLimit("gpt-4o-2024-08-06")).toBe(65536);
    expect(getModelTokenLimit("gpt-4o-mini-2024-07-18")).toBe(65536);
    expect(getModelTokenLimit("gpt-4-0613")).toBe(8192);
    expect(getModelTokenLimit("gpt-3.5-turbo-0125")).toBe(16385);
  });

  it("should return default limits for model families", () => {
    // Default for model families
    expect(getModelTokenLimit("gpt-4o-future-version")).toBe(65536);
    expect(getModelTokenLimit("gpt-4-future-version")).toBe(8192);
    expect(getModelTokenLimit("gpt-3.5-turbo-future-version")).toBe(16385);
  });

  it("should return conservative limit for unknown models", () => {
    expect(getModelTokenLimit("unknown-model")).toBe(4096);
  });
});

describe("convertToTiktokenModel", () => {
  it("should handle gpt-4 and gpt-4o models", () => {
    const messages: OpenAIMessage[] = [
      { role: "assistant" as OpenAIRole, content: "Test" },
    ];
    expect(() =>
      validateMessageLength(
        messages,
        "gpt-4o-2024-08-06",
        getEncodingForModel("gpt-4o-2024-08-06"),
      ),
    ).not.toThrow();
    expect(() =>
      validateMessageLength(
        messages,
        "gpt-4o-mini-2024-07-18",
        getEncodingForModel("gpt-4o-mini-2024-07-18"),
      ),
    ).not.toThrow();
    expect(() =>
      validateMessageLength(
        messages,
        "gpt-4-0613",
        getEncodingForModel("gpt-4-0613"),
      ),
    ).not.toThrow();
  });

  it("should handle gpt-3.5-turbo models", () => {
    const messages: OpenAIMessage[] = [
      { role: "assistant" as OpenAIRole, content: "Test" },
    ];
    expect(() =>
      validateMessageLength(
        messages,
        "gpt-4-0613",
        getEncodingForModel("gpt-4-0613"),
      ),
    ).not.toThrow();
  });

  it("should throw error for unsupported models", () => {
    const model = "unsupported-model";
    const messages: OpenAIMessage[] = [
      { role: "assistant" as OpenAIRole, content: "Test" },
    ];
    expect(() =>
      validateMessageLength(messages, model, getEncodingForModel(model)),
    ).toThrow(
      `Unsupported model: ${model}. Supported models are: ${CURRENT_MODEL_VERSIONS.join(", ")}`,
    );
  });
});

describe("validateMessageLength", () => {
  it("should validate messages within token limit using model-specific limits", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    // Test with gpt-4o-2024-08-06 (65536 token limit)
    const result1 = validateMessageLength(
      messages,
      "gpt-4o-2024-08-06",
      getEncodingForModel("gpt-4o-2024-08-06"),
    );
    expect(result1.isValid).toBe(true);
    expect(result1.tokenCount).toBeLessThan(
      MODEL_TOKEN_LIMITS["gpt-4o-2024-08-06"],
    );

    // Test with gpt-3.5-turbo-0125 (16385 token limit)
    const result2 = validateMessageLength(
      messages,
      "gpt-3.5-turbo-0125",
      getEncodingForModel("gpt-3.5-turbo-0125"),
    );
    expect(result2.isValid).toBe(true);
    expect(result2.tokenCount).toBeLessThan(
      MODEL_TOKEN_LIMITS["gpt-3.5-turbo-0125"],
    );
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

    const result = validateMessageLength(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(true);
  });

  it("should validate custom token limit", () => {
    const messages: OpenAIMessage[] = [
      { role: "system" as OpenAIRole, content: "You are a helpful assistant." },
      { role: "user" as OpenAIRole, content: "Hello!" },
    ];

    const result = validateMessageLength(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
      10,
    );
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
    expect(result.unrecognizedRoleCount).toBe(0);
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
    expect(result.unrecognizedRoleCount).toBe(0);
  });

  it("should detect unrecognized roles", () => {
    const messages: OpenAIMessage[] = [
      { role: "user" as OpenAIRole, content: "Hello!" },
      { role: "assistant" as OpenAIRole, content: "Hi there!" },
      { role: "invalid_role" as OpenAIRole, content: "This role is not valid" },
    ];

    const result = validateMessageRoles(messages);
    expect(result.isValid).toBe(false);
    expect(result.unrecognizedRoleCount).toBe(1);
  });

  it("should detect missing required and recommended roles", () => {
    const testCases = [
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
          { role: "assistant" as OpenAIRole, content: "Hi there!" },
        ],
        missing: [],
        isValid: true,
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
        isValid: false, // assistant is required
      },
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
          {
            role: "user" as OpenAIRole,
            content: "Helo!",
          },
        ],
        missing: ["assistant"],
        isValid: false, // assistant is required
      },
    ];

    for (const testCase of testCases) {
      const result = validateMessageRoles(testCase.messages);
      expect(result.isValid).toBe(testCase.isValid);
      for (const role of testCase.missing) {
        expect(result.missingRoles).toContain(role);
      }
    }
  });
});

describe("countAssistantTokens", () => {
  it("should count tokens in assistant messages only", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "system" as OpenAIRole,
        content: "You are a helpful assistant.",
      },
      { role: "user" as OpenAIRole, content: "Hello, how are you?" },
      {
        role: "assistant" as OpenAIRole,
        content: "I'm doing well, thank you for asking!",
      },
    ];

    const assistantTokens = countAssistantTokens(
      messages,
      getEncodingForModel("gpt-4-0613"),
    );
    expect(assistantTokens).toBeGreaterThan(0);

    // Only assistant message should be counted
    const totalTokens = validateMessageLength(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    ).tokenCount;
    expect(assistantTokens).toBeLessThan(totalTokens);
  });

  it("should handle messages with no assistant content", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "system" as OpenAIRole,
        content: "You are a helpful assistant.",
      },
      { role: "user" as OpenAIRole, content: "Hello!" },
    ];

    const assistantTokens = countAssistantTokens(
      messages,
      getEncodingForModel("gpt-4-0613"),
    );
    expect(assistantTokens).toBe(0);
  });
});

describe("calculateDistribution", () => {
  it("should calculate distribution statistics correctly", () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const distribution = calculateDistribution(values);

    expect(distribution.min).toBe(1);
    expect(distribution.max).toBe(10);
    expect(distribution.mean).toBe(5.5);
    expect(distribution.median).toBe(5.5);
    expect(distribution.p5).toBe(1); // 5th percentile
    expect(distribution.p95).toBe(10); // 95th percentile
  });

  it("should handle empty arrays", () => {
    const distribution = calculateDistribution([]);

    expect(distribution.min).toBe(0);
    expect(distribution.max).toBe(0);
    expect(distribution.mean).toBe(0);
    expect(distribution.median).toBe(0);
    expect(distribution.p5).toBe(0);
    expect(distribution.p95).toBe(0);
  });
});

describe("analyzeDataset", () => {
  it("should analyze dataset and provide statistics", () => {
    const dataset = [
      {
        messages: [
          {
            role: "system" as OpenAIRole,
            content: "You are a helpful assistant.",
          },
          { role: "user" as OpenAIRole, content: "Hello!" },
          {
            role: "assistant" as OpenAIRole,
            content: "Hi there! How can I help you today?",
          },
        ],
      },
      {
        messages: [
          { role: "user" as OpenAIRole, content: "What's the weather like?" },
          {
            role: "assistant" as OpenAIRole,
            content: "I don't have real-time weather data.",
          },
        ],
      },
      {
        messages: [
          { role: "user" as OpenAIRole, content: "Tell me a joke." },
          {
            role: "assistant" as OpenAIRole,
            content:
              "Why did the chicken cross the road? To get to the other side!",
          },
        ],
      },
    ];

    const analysis = analyzeDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );

    expect(analysis.missingSystemCount).toBe(2);
    expect(analysis.missingUserCount).toBe(0);
    expect(analysis.messageCounts.min).toBe(2);
    expect(analysis.messageCounts.max).toBe(3);
    expect(analysis.tokenCounts.min).toBeGreaterThan(0);
    expect(analysis.assistantTokenCounts.min).toBeGreaterThan(0);
    expect(analysis.tooLongCount).toBe(0);
  });

  it("should use model-specific token limits for validation", () => {
    const dataset = [
      {
        messages: [
          { role: "user" as OpenAIRole, content: "Hello!" },
          { role: "assistant" as OpenAIRole, content: "Hi there!" },
        ],
      },
    ];

    // Test with different models
    const analysis1 = analyzeDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    const analysis2 = analyzeDataset(
      dataset,
      "gpt-4o-2024-08-06",
      getEncodingForModel("gpt-4o-2024-08-06"),
    );

    // Both should pass with the small test messages
    expect(analysis1.tooLongCount).toBe(0);
    expect(analysis2.tooLongCount).toBe(0);

    // But they should use different token limits
    expect(getModelTokenLimit("gpt-4-0613")).toBe(8192);
    expect(getModelTokenLimit("gpt-4o-2024-08-06")).toBe(65536);
  });

  it("should detect examples exceeding token limit", () => {
    const dataset = [
      {
        messages: [
          { role: "user" as OpenAIRole, content: "Hello!" },
          { role: "assistant" as OpenAIRole, content: "Hi there!" },
        ],
      },
    ];

    // Set a very low token limit to force examples to exceed it
    const analysis = analyzeDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
      5,
    );

    expect(analysis.tooLongCount).toBe(1);
  });
});

describe("validateDataFormat", () => {
  it("should validate valid dataset entry", () => {
    const entry = {
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
        { role: "assistant", content: "Hi there! How can I help you today?" },
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(true);
    expect(Object.values(result.errors).every((count) => count === 0)).toBe(
      true,
    );
  });

  it("should detect invalid data type", () => {
    const result = validateDataFormat("not an object");
    expect(result.isValid).toBe(false);
    expect(result.errors.data_type).toBe(1);
  });

  it("should detect missing messages list", () => {
    const result = validateDataFormat({});
    expect(result.isValid).toBe(false);
    expect(result.errors.missing_messages_list).toBe(1);
  });

  it("should detect message missing key", () => {
    const entry = {
      messages: [
        { role: "user", content: "Hello!" },
        { role: "assistant" }, // Missing content
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(false);
    expect(result.errors.message_missing_key).toBe(1);
  });

  it("should detect unrecognized keys in messages", () => {
    const entry = {
      messages: [
        { role: "user", content: "Hello!" },
        { role: "assistant", content: "Hi!", unknown_key: "value" },
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(false);
    expect(result.errors.message_unrecognized_key).toBe(1);
  });

  it("should detect unrecognized role", () => {
    const entry = {
      messages: [
        { role: "user", content: "Hello!" },
        { role: "invalid_role", content: "This is not a valid role" },
        { role: "assistant", content: "Hi!" },
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(false);
    expect(result.errors.unrecognized_role).toBe(1);
  });

  it("should detect missing content", () => {
    const entry = {
      messages: [
        { role: "user", content: "Hello!" },
        { role: "assistant", content: 123 },
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(false);
    expect(result.errors.missing_content).toBe(1);
  });

  it("should detect missing assistant message", () => {
    const entry = {
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
      ],
    };

    const result = validateDataFormat(entry);
    expect(result.isValid).toBe(false);
    expect(result.errors.example_missing_assistant_message).toBe(1);
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

    const result = validateMessage(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(true);
    expect(result.lengthValidation.isValid).toBe(true);
    expect(result.rolesValidation.isValid).toBe(true);
    expect(result.formatValidation.isValid).toBe(true);
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

    const result = validateMessage(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
      10,
    );
    expect(result.isValid).toBe(false);
    expect(result.lengthValidation.isValid).toBe(false);
    expect(result.rolesValidation.isValid).toBe(true);
    expect(result.formatValidation.isValid).toBe(true);
  });

  it("should validate messages with missing recommended roles", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "assistant" as OpenAIRole,
        content: "Hi there! How can I help you today?",
      },
    ];

    const result = validateMessage(
      messages,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(true); // Should be valid since assistant is present
    expect(result.lengthValidation.isValid).toBe(true);
    expect(result.rolesValidation.isValid).toBe(true); // Should be true since only assistant is required
    expect(result.formatValidation.isValid).toBe(true); // Should be true since example_missing_user_message is not considered for validity
    expect(
      result.formatValidation.errors.example_missing_assistant_message,
    ).toBe(0);
  });
});

describe("validateDataset", () => {
  it("should validate invalid dataset", () => {
    const dataset = [
      {
        messages: [{ role: "assistant", content: "1" }],
      },
      {
        messages: [{ role: "assistant", content: "2" }],
      },
    ];

    const result = validateDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(false);
    expect(result.invalidEntries).toBe(0);
    expect(result.errorCounts.insufficient_examples).toBe(1);
  });

  it("should validate valid dataset", () => {
    const dataset = [
      {
        messages: [{ role: "assistant", content: "1" }],
      },
      {
        messages: [{ role: "assistant", content: "2" }],
      },
      {
        messages: [{ role: "assistant", content: "3" }],
      },
      {
        messages: [{ role: "assistant", content: "4" }],
      },
      {
        messages: [{ role: "assistant", content: "5" }],
      },
      {
        messages: [{ role: "assistant", content: "6" }],
      },
      {
        messages: [{ role: "assistant", content: "7" }],
      },
      {
        messages: [{ role: "assistant", content: "8" }],
      },
      {
        messages: [{ role: "assistant", content: "9" }],
      },
      {
        messages: [{ role: "assistant", content: "10" }],
      },
    ];

    const result = validateDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(true);
    expect(result.invalidEntries).toBe(0);
    expect(result.errorCounts.insufficient_examples).toBe(0);
    expect(
      Object.values(result.errorCounts).every((count) => count === 0),
    ).toBe(true);
  });

  it("should detect invalid entries in dataset", () => {
    const dataset = [
      {
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello!" },
          { role: "assistant", content: "Hi there! How can I help you today?" },
        ],
      },
      "not an object", // Invalid entry
      {
        messages: [
          { role: "user", content: "What's the weather like?" },
          // Missing assistant message
        ],
      },
      {
        messages: [{ role: "assistant", content: "1" }],
      },
      {
        messages: [{ role: "assistant", content: "2" }],
      },
      {
        messages: [{ role: "assistant", content: "3" }],
      },
      {
        messages: [{ role: "assistant", content: "4" }],
      },
      {
        messages: [{ role: "assistant", content: "5" }],
      },
      {
        messages: [{ role: "assistant", content: "6" }],
      },
      {
        messages: [{ role: "assistant", content: "7" }],
      },
      {
        messages: [{ role: "assistant", content: "8" }],
      },
      {
        messages: [{ role: "assistant", content: "9" }],
      },
      {
        messages: [{ role: "assistant", content: "10" }],
      },
    ];

    const result = validateDataset(
      dataset,
      "gpt-4-0613",
      getEncodingForModel("gpt-4-0613"),
    );
    expect(result.isValid).toBe(false);
    expect(result.invalidEntries).toBe(2);
    expect(result.errorCounts.data_type).toBe(1);
    expect(result.errorCounts.example_missing_assistant_message).toBe(1);
  });
});
