import { describe, it, expect } from "vitest";
import OpenAI from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionFunctionMessageParam,
  ChatCompletionTool,
} from "openai/resources/chat/completions";
import {
  getTokensFromMessages,
  getTokensForTools,
  countAssistantTokens,
  getSimpleTokenCount,
  getModelTokenLimit,
  getEncodingForModel,
} from "./openAITokenCounter";
import { CURRENT_MODEL_VERSIONS } from "./constants";
import type { OpenAIMessage, ToolFunction } from "./types";

const shouldRunApiTests =
  !!process.env.OPENAI_API_KEY &&
  process.env.ENABLE_TEST_WITH_OPENAI_API === "true";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

describe("openAITokenCounter", () => {
  describe("getTokensFromMessages", () => {
    it("should count tokens for basic messages", () => {
      const messages: OpenAIMessage[] = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
        { role: "assistant", content: "Hi there! How can I help you today?" },
      ];
      const count = getTokensFromMessages(
        messages,
        "gpt-3.5-turbo",
        getEncodingForModel("gpt-3.5-turbo"),
      );
      expect(count).toBeGreaterThan(0);
    });

    it("should handle messages with name field", () => {
      const messages: OpenAIMessage[] = [
        { role: "assistant", content: "Hello", name: "AI" },
      ];
      const count = getTokensFromMessages(
        messages,
        "gpt-4",
        getEncodingForModel("gpt-4"),
      );
      expect(count).toBeGreaterThan(4); // base(3) + content + name + name_tax(1)
    });

    it("should handle empty messages", () => {
      const messages: OpenAIMessage[] = [];
      const count = getTokensFromMessages(
        messages,
        "gpt-3.5-turbo",
        getEncodingForModel("gpt-3.5-turbo"),
      );
      expect(count).toBe(3); // priming tokens
    });
  });

  describe("getTokensForTools", () => {
    const sampleFunction: ToolFunction = {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get the current weather",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "City name",
            },
            unit: {
              type: "string",
              description: "Temperature unit",
              enum: ["celsius", "fahrenheit"],
            },
          },
        },
      },
    };

    it("should count tokens for functions and messages", () => {
      const messages: OpenAIMessage[] = [
        { role: "user", content: "What's the weather?" },
      ];
      const count = getTokensForTools(
        [sampleFunction],
        messages,
        "gpt-4",
        getEncodingForModel("gpt-4"),
      );
      expect(count).toBeGreaterThan(0);
    });

    it("should handle empty functions array", () => {
      const messages: OpenAIMessage[] = [{ role: "user", content: "Hello" }];
      const count = getTokensForTools(
        [],
        messages,
        "gpt-4",
        getEncodingForModel("gpt-4"),
      );
      expect(count).toBeGreaterThan(0);
    });
  });

  describe("countAssistantTokens", () => {
    it("should count only assistant messages", () => {
      const messages: OpenAIMessage[] = [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
        { role: "user", content: "How are you?" },
        { role: "assistant", content: "I'm doing well, thanks!" },
      ];
      const count = countAssistantTokens(
        messages,
        getEncodingForModel("gpt-3.5-turbo"),
      );
      expect(count).toBeGreaterThan(0);
    });

    it("should handle assistant messages with function calls", () => {
      const messages: OpenAIMessage[] = [
        {
          role: "assistant",
          function_call: {
            name: "get_weather",
            arguments: '{"location":"Tokyo","unit":"celsius"}',
          },
        },
      ];
      const count = countAssistantTokens(
        messages,
        getEncodingForModel("gpt-4"),
      );
      expect(count).toBeGreaterThan(3); // base tokens + function call tokens
    });

    it("should handle assistant messages with tool calls", () => {
      const messages: OpenAIMessage[] = [
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"location":"Tokyo","unit":"celsius"}',
              },
            },
          ],
        },
      ];
      const count = countAssistantTokens(
        messages,
        getEncodingForModel("gpt-4"),
      );
      expect(count).toBeGreaterThan(3); // base tokens + tool call tokens
    });
  });

  describe("getSimpleTokenCount", () => {
    it("should provide rough token estimation", () => {
      const messages: OpenAIMessage[] = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
        { role: "assistant", content: "Hi there! How can I help you today?" },
      ];
      const count = getSimpleTokenCount(messages);
      expect(count).toBeGreaterThan(0);
    });

    it("should handle custom tokens per message", () => {
      const messages: OpenAIMessage[] = [{ role: "user", content: "Hello" }];
      const count = getSimpleTokenCount(messages, 4, 2);
      expect(count).toBeGreaterThan(7); // 4 (base) + content tokens + 3 (priming)
    });

    it("should handle messages with name field", () => {
      const messages: OpenAIMessage[] = [
        { role: "assistant", content: "Hello", name: "AI" },
      ];
      const count = getSimpleTokenCount(messages);
      expect(count).toBeGreaterThan(4); // base(3) + content + name + name_tax(1)
    });
  });

  describe("getModelTokenLimit", () => {
    it("should return correct limits for known models", () => {
      expect(getModelTokenLimit("gpt-4-0613")).toBe(8192);
      expect(getModelTokenLimit("gpt-3.5-turbo-0125")).toBe(16385);
    });

    it("should handle model variants", () => {
      expect(getModelTokenLimit("gpt-4o-2024-08-06")).toBe(65536);
      expect(getModelTokenLimit("gpt-4-0613")).toBe(8192);
      expect(getModelTokenLimit("gpt-3.5-turbo-1106")).toBe(16385);
    });

    it("should return default limit for unknown models", () => {
      expect(getModelTokenLimit("unknown-model")).toBe(4096);
    });
  });

  describe("API Token Count Verification", () => {
    const example_messages: OpenAIMessage[] = [
      {
        role: "system",
        content:
          "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
      },
      {
        role: "user",
        name: "example_user",
        content: "New synergies will help drive top-line growth.",
      },
      {
        role: "assistant",
        name: "example_assistant",
        content: "Things working well together will increase revenue.",
      },
      {
        role: "user",
        name: "example_user",
        content:
          "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
      },
      {
        role: "assistant",
        name: "example_assistant",
        content:
          "Let's talk later when we're less busy about how to do better.",
      },
      {
        role: "user",
        content:
          "This late pivot means we don't have time to boil the ocean for the client deliverable.",
      },
    ];

    it.skipIf(!shouldRunApiTests).each([...CURRENT_MODEL_VERSIONS])(
      "should match OpenAI API token count for %s",
      async (model) => {
        // Calculate tokens using our function
        const calculatedTokens = getTokensFromMessages(
          example_messages,
          model,
          getEncodingForModel(model),
        );

        // Convert messages to OpenAI API format
        const apiMessages: ChatCompletionMessageParam[] = example_messages.map(
          (msg) => {
            // Ensure content is always a string
            const content = msg.content || "";

            switch (msg.role) {
              case "system":
                return {
                  role: "system",
                  content,
                  name: msg.name,
                } as ChatCompletionSystemMessageParam;
              case "user":
                return {
                  role: "user",
                  content,
                  name: msg.name,
                } as ChatCompletionUserMessageParam;
              case "assistant":
                return {
                  role: "assistant",
                  content,
                  name: msg.name,
                  function_call: msg.function_call,
                  tool_calls: msg.tool_calls,
                } as ChatCompletionAssistantMessageParam;
              case "function":
                return {
                  role: "function",
                  content,
                  name: msg.name,
                } as ChatCompletionFunctionMessageParam;
              case "tool":
                return {
                  role: "tool",
                  content,
                  name: msg.name,
                  tool_call_id: msg.tool_call_id || "",
                } as ChatCompletionToolMessageParam;
              default:
                throw new Error(`Unsupported role: ${msg.role}`);
            }
          },
        );

        // Get token count from OpenAI API
        const response = await client.chat.completions.create({
          model,
          messages: apiMessages,
          temperature: 0,
          max_tokens: 1,
        });

        expect(calculatedTokens).toBe(response.usage?.prompt_tokens ?? 0);
      },
    );
  });

  describe("API Token Count with Tools Verification", () => {
    const example_tools = [
      {
        type: "function" as const,
        function: {
          name: "get_current_weather",
          description: "Get the current weather in a given location",
          parameters: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city and state, e.g. San Francisco, CA",
              },
              unit: {
                type: "string",
                description: "The unit of temperature to return",
                enum: ["celsius", "fahrenheit"],
              },
            },
          },
        },
      },
    ];

    const example_messages: OpenAIMessage[] = [
      {
        role: "system",
        content:
          "You are a helpful assistant that can answer to questions about the weather.",
      },
      {
        role: "user",
        content: "What's the weather like in San Francisco?",
      },
    ];

    it.skipIf(!shouldRunApiTests).each([...CURRENT_MODEL_VERSIONS])(
      "should match OpenAI API token count with tools for %s",
      async (model) => {
        // Calculate tokens using our function
        const calculatedTokens = getTokensForTools(
          example_tools,
          example_messages,
          model,
          getEncodingForModel(model),
        );

        // Convert messages to OpenAI API format
        const apiMessages: ChatCompletionMessageParam[] = example_messages.map(
          (msg) => {
            const content = msg.content || "";
            switch (msg.role) {
              case "system":
                return {
                  role: "system",
                  content,
                  name: msg.name,
                } as ChatCompletionSystemMessageParam;
              case "user":
                return {
                  role: "user",
                  content,
                  name: msg.name,
                } as ChatCompletionUserMessageParam;
              default:
                throw new Error(`Unsupported role: ${msg.role}`);
            }
          },
        );

        // Get token count from OpenAI API
        const response = await client.chat.completions.create({
          model,
          messages: apiMessages,
          tools: example_tools as ChatCompletionTool[],
          temperature: 0,
          max_tokens: 100,
        });

        const apiTokens = response.usage?.prompt_tokens ?? 0;

        expect(calculatedTokens).toBe(apiTokens);
      },
    );
  });
});
