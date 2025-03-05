import { encoding_for_model } from "tiktoken";
import type { TiktokenModel } from "tiktoken";
import type { OpenAIMessage } from "./types";

/**
 * Convert OpenAI model names to tiktoken model names
 * Reference: https://platform.openai.com/docs/guides/fine-tuning/token-counting
 *
 * Supported fine-tuning models as of 20250304:
 * - gpt-4o-2024-08-06
 * - gpt-4o-mini-2024-07-18
 * - gpt-4-0613
 * - gpt-3.5-turbo-0125
 * - gpt-3.5-turbo-1106
 * - gpt-3.5-turbo-0613
 */
function convertToTiktokenModel(model: string): TiktokenModel {
  if (model.startsWith("gpt-4o-") || model.startsWith("gpt-4-")) {
    return "gpt-4";
  }
  if (model.startsWith("gpt-3.5-turbo")) {
    return "gpt-3.5-turbo";
  }

  throw new Error(
    `Unsupported model: ${model}. Supported models are: gpt-4o-*, gpt-4-*, gpt-3.5-turbo*`,
  );
}

/**
 * Validates the total length of messages using tiktoken
 * @param messages Array of messages to validate
 * @param model Model name to use for token counting
 * @param maxTokens Maximum allowed tokens for all messages
 * @returns Object containing validation result and token count
 */
export function validateMessageLength(
  messages: OpenAIMessage[],
  model: string,
  maxTokens: number = 4096,
): { isValid: boolean; tokenCount: number } {
  const enc = encoding_for_model(convertToTiktokenModel(model));

  let totalTokens = 0;
  for (const message of messages) {
    if (message.content) {
      totalTokens += enc.encode(message.content).length;
    }
    // Add tokens for message format overhead:
    // - 3 tokens at the start of the message
    // - 1 token at the end of the message
    totalTokens += 4;
  }

  enc.free(); // Free up memory

  return {
    isValid: totalTokens <= maxTokens,
    tokenCount: totalTokens,
  };
}

/**
 * Validates that the messages contain required roles (user, assistant)
 * @param messages Array of messages to validate
 * @returns Object containing validation result and any missing roles
 */
export function validateMessageRoles(messages: OpenAIMessage[]): {
  isValid: boolean;
  missingRoles: string[];
} {
  const roles = new Set(messages.map((m) => m.role));
  const missingRoles: string[] = [];
  const requiredRoles = ["user", "assistant"] as const;

  for (const role of requiredRoles) {
    if (!roles.has(role)) {
      missingRoles.push(role);
    }
  }

  return {
    isValid: missingRoles.length === 0,
    missingRoles,
  };
}

/**
 * Validates messages against all validation rules
 * @param messages Array of messages to validate
 * @param model Model name for token counting
 * @param maxTokens Maximum allowed tokens
 * @returns Validation result with details
 */
export function validateMessage(
  messages: OpenAIMessage[],
  model: string,
  maxTokens?: number,
): {
  isValid: boolean;
  lengthValidation: { isValid: boolean; tokenCount: number };
  rolesValidation: { isValid: boolean; missingRoles: string[] };
} {
  const lengthValidation = validateMessageLength(messages, model, maxTokens);
  const rolesValidation = validateMessageRoles(messages);

  return {
    isValid: lengthValidation.isValid && rolesValidation.isValid,
    lengthValidation,
    rolesValidation,
  };
}
