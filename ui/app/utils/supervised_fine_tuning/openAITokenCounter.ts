// TypeScript port of the token counting example using Tiktoken from the Python version of OpenAI Cookbook
// Reference: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

import { encoding_for_model, get_encoding } from "tiktoken";
import type { Tiktoken, TiktokenModel } from "tiktoken";
import { MODEL_TOKEN_LIMITS, CURRENT_MODEL_VERSIONS } from "./constants";
import type { OpenAIMessage, ToolFunction } from "./types";
import { logger } from "~/utils/logger";

/**
 * Converts model name to tiktoken model name
 * @param model Model name to convert
 * @returns Corresponding tiktoken model name
 */
export function convertToTiktokenModel(
  model: string,
): TiktokenModel | undefined {
  // Check if the model is already a valid TiktokenModel
  try {
    // Try to create an encoder with the model name
    const encoder = encoding_for_model(model as TiktokenModel);
    encoder.free();
    return model as TiktokenModel;
  } catch {
    // If the model is not valid, proceed with model family checks
  }

  // Handle model families
  if (model.startsWith("gpt-4o-")) {
    return "gpt-4";
  }
  if (model.startsWith("gpt-4-")) {
    return "gpt-4";
  }
  if (model.startsWith("gpt-3.5-turbo")) {
    return "gpt-3.5-turbo";
  }

  return;
}

/**
 * Gets token limit for a given model
 * @param model Model name
 * @returns Token limit for the model
 */
export function getModelTokenLimit(model: string): number {
  // Check exact model match first
  if (model in MODEL_TOKEN_LIMITS) {
    return MODEL_TOKEN_LIMITS[model];
  }

  // Handle model families
  if (model.startsWith("gpt-4o-")) {
    return MODEL_TOKEN_LIMITS["gpt-4o-2024-08-06"];
  }
  if (model.startsWith("gpt-4-")) {
    return MODEL_TOKEN_LIMITS["gpt-4-0613"];
  }
  if (model.startsWith("gpt-3.5-turbo")) {
    return MODEL_TOKEN_LIMITS["gpt-3.5-turbo-0125"];
  }

  // Default to conservative limit for unknown models
  return 4096;
}

/**
 * Gets encoding for a given model.
 * Following OpenAI's implementation, we use o200k_base encoding as a fallback for unknown models.
 * https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken#6-counting-tokens-for-chat-completions-api-calls
 * @param model Model name
 * @returns Tiktoken encoding
 */
export function getEncodingForModel(model: string) {
  const tiktokenModel = convertToTiktokenModel(model);
  if (!tiktokenModel) {
    logger.warn(`Unknown model: ${model}, using o200k_base`);
    return get_encoding("o200k_base");
  }
  return encoding_for_model(tiktokenModel);
}

/**
 * Calculate the number of tokens used by a list of messages
 * @param messages Array of messages to count tokens for
 * @param model Model name to use for token counting
 * @returns Number of tokens used by the messages
 */
export function getTokensFromMessages(
  messages: OpenAIMessage[],
  model: string,
  enc: Tiktoken,
): number {
  const tokens_per_message = 3;
  const tokens_per_name = 1;

  // Check if the model is a supported version
  if (
    !CURRENT_MODEL_VERSIONS.includes(
      model as (typeof CURRENT_MODEL_VERSIONS)[number],
    )
  ) {
    logger.warn(
      `Warning: ${model} is not a current version. Using default token counts.`,
    );
  }

  let num_tokens = 0;
  for (const message of messages) {
    num_tokens += tokens_per_message;
    for (const [key, value] of Object.entries(message)) {
      if (value) {
        num_tokens += enc.encode(String(value)).length;
        if (key === "name") {
          num_tokens += tokens_per_name;
        }
      }
    }
  }
  num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>

  return num_tokens;
}

/**
 * Calculate the number of tokens used by messages and tools
 * @param functions Array of function definitions
 * @param messages Array of messages
 * @param model Model name for token counting
 * @returns Total number of tokens
 */
export function getTokensForTools(
  functions: ToolFunction[],
  messages: OpenAIMessage[],
  model: string,
  enc: Tiktoken,
): number {
  let func_init = 0;
  let prop_init = 0;
  let prop_key = 0;
  let enum_init = 0;
  let enum_item = 0;
  let func_end = 0;

  if (model.startsWith("gpt-4o") || model === "gpt-4o-mini") {
    func_init = 7;
    prop_init = 3;
    prop_key = 3;
    enum_init = -3;
    enum_item = 3;
    func_end = 12;
  } else if (model.startsWith("gpt-3.5-turbo") || model.startsWith("gpt-4")) {
    func_init = 10;
    prop_init = 3;
    prop_key = 3;
    enum_init = -3;
    enum_item = 3;
    func_end = 12;
  } else {
    throw new Error(
      `getTokensForTools() is not implemented for model ${model}`,
    );
  }

  let func_token_count = 0;
  if (functions.length > 0) {
    for (const f of functions) {
      func_token_count += func_init; // Add tokens for start of each function
      const function_def = f.function;
      const f_name = function_def.name;
      let f_desc = function_def.description;
      if (f_desc.endsWith(".")) {
        f_desc = f_desc.slice(0, -1);
      }
      const line = `${f_name}:${f_desc}`;
      func_token_count += enc.encode(line).length; // Add tokens for name and description

      if (Object.keys(function_def.parameters.properties).length > 0) {
        func_token_count += prop_init; // Add tokens for start of properties
        for (const [key, value] of Object.entries(
          function_def.parameters.properties,
        )) {
          func_token_count += prop_key; // Add tokens for each property
          const p_name = key;
          const p_type = value.type;
          let p_desc = value.description;
          if (p_desc.endsWith(".")) {
            p_desc = p_desc.slice(0, -1);
          }

          if (value.enum) {
            func_token_count += enum_init; // Add tokens for enum list
            for (const item of value.enum) {
              func_token_count += enum_item;
              func_token_count += enc.encode(item).length;
            }
          }

          const prop_line = `${p_name}:${p_type}:${p_desc}`;
          func_token_count += enc.encode(prop_line).length;
        }
      }
    }
    func_token_count += func_end;
  }

  const messagesTokenCount = getTokensFromMessages(messages, model, enc);
  return messagesTokenCount + func_token_count;
}

/**
 * Calculates the number of tokens in assistant messages only
 * @param messages Array of messages
 * @param model Model name for token counting
 * @returns Number of tokens in assistant messages
 */
export function countAssistantTokens(
  messages: OpenAIMessage[],
  enc: Tiktoken,
): number {
  let assistantTokens = 0;
  for (const message of messages) {
    if (message.role === "assistant") {
      // Add base tokens for assistant message
      assistantTokens += 3; // tokens_per_message

      // Add tokens for content if present
      if (message.content) {
        assistantTokens += enc.encode(message.content).length;
      }

      // Add tokens for name if present
      if (message.name) {
        assistantTokens += enc.encode(message.name).length + 1; // tokens_per_name
      }

      // Add tokens for function calls if present
      if (message.function_call) {
        const functionCall = message.function_call;
        assistantTokens += enc.encode(functionCall.name).length;
        assistantTokens += enc.encode(functionCall.arguments).length;
      }

      // Add tokens for tool calls if present
      if (message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          assistantTokens += enc.encode(toolCall.id).length;
          assistantTokens += enc.encode(toolCall.type).length;
          assistantTokens += enc.encode(toolCall.function.name).length;
          assistantTokens += enc.encode(toolCall.function.arguments).length;
        }
      }
    }
  }

  return assistantTokens;
}

/**
 * Simple token count estimation without using tiktoken
 * Less accurate but faster and doesn't require tiktoken dependency
 * @param messages Array of messages to count tokens for
 * @param tokens_per_message Base tokens per message (default: 3)
 * @param tokens_per_name Additional tokens for name field (default: 1)
 * @returns Estimated number of tokens
 */
export function getSimpleTokenCount(
  messages: OpenAIMessage[],
  tokens_per_message = 3,
  tokens_per_name = 1,
): number {
  let num_tokens = 0;
  for (const message of messages) {
    num_tokens += tokens_per_message;
    for (const [key, value] of Object.entries(message)) {
      if (value) {
        // Rough estimation: 1 token ≈ 4 characters
        const str = String(value);
        num_tokens += Math.ceil(str.length / 4);
        if (key === "name") {
          num_tokens += tokens_per_name;
        }
      }
    }
  }
  num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
  return num_tokens;
}
