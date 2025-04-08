/**
 * OpenAI Supervised Fine-Tuning Validation Module
 *
 * This module provides validation utilities for OpenAI's supervised fine-tuning datasets.
 * It includes functions to validate message formats, token lengths, and role requirements
 * according to OpenAI's specifications.
 *
 * Public interface:
 * - validateMessage: Comprehensive validation of messages against all rules
 * - analyzeDataset: Provides statistical analysis of a dataset
 * - validateDataset: Validates an entire dataset against all rules
 */
import type { OpenAIMessage, Distribution } from "./types";
import {
  getModelTokenLimit,
  getTokensFromMessages,
  countAssistantTokens,
} from "./openAITokenCounter";
import {
  OPENAI_ROLES,
  REQUIRED_ROLES,
  RECOMMENDED_ROLES,
  VALID_MESSAGE_KEYS,
  CURRENT_MODEL_VERSIONS,
} from "./constants";
import type { Tiktoken } from "tiktoken";

/**
 * Format error types based on Python implementation
 */
type FormatErrorType =
  | "data_type"
  | "missing_messages_list"
  | "message_missing_key"
  | "message_unrecognized_key"
  | "unrecognized_role"
  | "missing_content"
  | "example_missing_assistant_message"
  | "insufficient_examples";

/**
 * Validates the total length of messages using tiktoken for OpenAI models
 * @param messages Array of messages to validate
 * @param model OpenAI model name to use for token counting
 * @param maxTokens Maximum allowed tokens for all messages (default: uses model-specific limit)
 * @returns Object containing validation result and token count
 *
 * Note: This validation is specific to OpenAI models and uses OpenAI's token counting logic.
 * It relies on the model-specific token limits defined in constants.ts.
 */
export function validateMessageLength(
  messages: OpenAIMessage[],
  model: string,
  enc: Tiktoken,
  maxTokens?: number,
): { isValid: boolean; tokenCount: number } {
  // Validate model version
  const isValidModel = CURRENT_MODEL_VERSIONS.includes(
    model as (typeof CURRENT_MODEL_VERSIONS)[number],
  );
  if (!isValidModel) {
    throw new Error(
      `Unsupported model: ${model}. Supported models are: ${CURRENT_MODEL_VERSIONS.join(", ")}`,
    );
  }

  const tokenCount = getTokensFromMessages(messages, model, enc);
  const tokenLimit = maxTokens ?? getModelTokenLimit(model);

  return {
    isValid: tokenCount <= tokenLimit,
    tokenCount,
  };
}

/**
 * Validates that the messages contain required roles (assistant)
 * and checks for unrecognized roles
 * @param messages Array of messages to validate
 * @returns Object containing validation result, any missing roles, and unrecognized role count
 */
export function validateMessageRoles(messages: OpenAIMessage[]): {
  isValid: boolean;
  missingRoles: string[];
  unrecognizedRoleCount: number;
} {
  const roles = new Set(messages.map((m) => m.role));
  const missingRoles: string[] = [];
  let unrecognizedRoleCount = 0;

  // Check for required roles
  for (const role of REQUIRED_ROLES) {
    if (!roles.has(role)) {
      missingRoles.push(role);
    }
  }

  // Check for unrecognized roles
  for (const message of messages) {
    if (!OPENAI_ROLES.includes(message.role)) {
      unrecognizedRoleCount++;
    }
  }

  // Also track recommended roles for reporting purposes
  const missingRecommendedRoles: string[] = [];
  for (const role of RECOMMENDED_ROLES) {
    if (!roles.has(role)) {
      missingRecommendedRoles.push(role);
    }
  }

  return {
    isValid: missingRoles.length === 0 && unrecognizedRoleCount === 0,
    missingRoles: missingRoles,
    unrecognizedRoleCount,
  };
}

/**
 * Validates the format of a dataset entry
 * @param data The dataset entry to validate
 * @returns Object containing validation results and error counts
 */
export function validateDataFormat(data: unknown): {
  isValid: boolean;
  errors: Record<FormatErrorType, number>;
} {
  const errors: Record<FormatErrorType, number> = {
    data_type: 0,
    missing_messages_list: 0,
    message_missing_key: 0,
    message_unrecognized_key: 0,
    unrecognized_role: 0,
    missing_content: 0,
    example_missing_assistant_message: 0,
    insufficient_examples: 0,
  };

  // Data type check
  if (typeof data !== "object" || data === null) {
    errors.data_type++;
    return { isValid: false, errors };
  }

  // Presence of messages list
  const entry = data as Record<string, unknown>;
  if (!entry.messages || !Array.isArray(entry.messages)) {
    errors.missing_messages_list++;
    return { isValid: false, errors };
  }

  const messages = entry.messages as Record<string, unknown>[];
  let hasAssistantMessage = false;

  // Check each message
  for (const message of messages) {
    // Message keys check
    if (
      !message.role ||
      (message.content === undefined &&
        !message.function_call &&
        !message.tool_calls)
    ) {
      errors.message_missing_key++;
    }

    // Unrecognized keys in messages
    for (const key in message) {
      if (
        !VALID_MESSAGE_KEYS.includes(key as (typeof VALID_MESSAGE_KEYS)[number])
      ) {
        errors.message_unrecognized_key++;
        break;
      }
    }

    // Role validation
    const role = message.role;
    if (
      typeof role === "string" &&
      !OPENAI_ROLES.includes(role as (typeof OPENAI_ROLES)[number])
    ) {
      errors.unrecognized_role++;
    }

    // Content validation
    const content = message.content;
    const functionCall = message.function_call;
    const toolCalls = message.tool_calls;

    if (
      (content === undefined && !functionCall && !toolCalls) ||
      (content !== undefined && typeof content !== "string")
    ) {
      errors.missing_content++;
    }

    // Check for assistant message
    if (role === "assistant") {
      hasAssistantMessage = true;
    }
  }

  if (!hasAssistantMessage) {
    errors.example_missing_assistant_message++;
  }

  // Only consider certain errors for validity check
  const isValid = Object.values(errors).every((count) => count === 0);
  return { isValid, errors };
}

/**
 * Validates messages against all validation rules
 * @param messages Array of messages to validate
 * @param model Model name for token counting
 * @param maxTokens Maximum allowed tokens (optional, uses model-specific limit if not provided)
 * @returns Validation result with details
 */
export function validateMessage(
  messages: OpenAIMessage[],
  model: string,
  enc: Tiktoken,
  maxTokens?: number,
): {
  isValid: boolean;
  lengthValidation: { isValid: boolean; tokenCount: number };
  rolesValidation: {
    isValid: boolean;
    missingRoles: string[];
    unrecognizedRoleCount: number;
  };
  formatValidation: {
    isValid: boolean;
    errors: Record<FormatErrorType, number>;
  };
} {
  const lengthValidation = validateMessageLength(messages, model, enc, maxTokens);
  const rolesValidation = validateMessageRoles(messages);

  // Create a mock dataset entry for format validation
  const mockEntry = { messages };
  const formatValidation = validateDataFormat(mockEntry);

  return {
    isValid:
      lengthValidation.isValid &&
      rolesValidation.isValid &&
      formatValidation.isValid,
    lengthValidation,
    rolesValidation,
    formatValidation,
  };
}

/**
 * Analyzes dataset for warnings and token statistics
 * @param dataset Array of dataset entries
 * @param model Model name for token counting
 * @param maxTokens Maximum allowed tokens (optional, uses model-specific limit if not provided)
 * @returns Analysis results with warnings and token statistics
 */
export function analyzeDataset(
  dataset: { messages: OpenAIMessage[] }[],
  model: string,
  enc: Tiktoken,
  maxTokens?: number,
): {
  missingSystemCount: number;
  missingUserCount: number;
  messageCounts: Distribution;
  tokenCounts: Distribution;
  assistantTokenCounts: Distribution;
  tooLongCount: number;
} {
  const tokenLimit = maxTokens ?? getModelTokenLimit(model);
  let missingSystemCount = 0;
  let missingUserCount = 0;
  let tooLongCount = 0;

  const messageCounts: number[] = [];
  const tokenCounts: number[] = [];
  const assistantTokenCounts: number[] = [];

  for (const entry of dataset) {
    const { messages } = entry;

    // Check for missing system/user messages
    if (!messages.some((m) => m.role === "system")) {
      missingSystemCount++;
    }
    if (!messages.some((m) => m.role === "user")) {
      missingUserCount++;
    }

    messageCounts.push(messages.length);

    const totalTokens = validateMessageLength(messages, model, enc).tokenCount;
    tokenCounts.push(totalTokens);
    const assistantTokenCount = countAssistantTokens(messages, enc);
    assistantTokenCounts.push(assistantTokenCount);

    // Check token limit
    if (totalTokens > tokenLimit) {
      tooLongCount++;
    }
  }

  return {
    missingSystemCount,
    missingUserCount,
    messageCounts: calculateDistribution(messageCounts),
    tokenCounts: calculateDistribution(tokenCounts),
    assistantTokenCounts: calculateDistribution(assistantTokenCounts),
    tooLongCount,
  };
}

/**
 * Calculates statistical distribution of numeric values
 * @param values Array of numeric values
 * @returns Distribution statistics
 */
export function calculateDistribution(values: number[]): Distribution {
  if (values.length === 0) {
    return {
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
      p5: 0,
      p95: 0,
    };
  }

  // Sort values for percentile calculations
  const sortedValues = [...values].sort((a, b) => a - b);

  // Calculate mean
  const sum = sortedValues.reduce((acc, val) => acc + val, 0);
  const mean = sum / sortedValues.length;

  // Calculate median
  const midIndex = Math.floor(sortedValues.length / 2);
  const median =
    sortedValues.length % 2 === 0
      ? (sortedValues[midIndex - 1] + sortedValues[midIndex]) / 2
      : sortedValues[midIndex];

  // Calculate percentiles
  const p5Index = Math.floor(sortedValues.length * 0.05);
  const p95Index = Math.floor(sortedValues.length * 0.95);

  return {
    min: sortedValues[0],
    max: sortedValues[sortedValues.length - 1],
    mean,
    median,
    p5: sortedValues[p5Index],
    p95: sortedValues[p95Index],
  };
}

/**
 * Validates a complete dataset against all validation rules
 * @param dataset Array of dataset entries
 * @param model Model name for token counting
 * @param maxTokens Maximum allowed tokens (optional, uses model-specific limit if not provided)
 * @returns Validation result with details and error counts
 */
export function validateDataset(
  dataset: unknown[],
  model: string,
  enc: Tiktoken,
  maxTokens?: number,
): {
  isValid: boolean;
  errorCounts: Record<FormatErrorType, number>;
  invalidEntries: number;
} {
  const errorCounts: Record<FormatErrorType, number> = {
    data_type: 0,
    missing_messages_list: 0,
    message_missing_key: 0,
    message_unrecognized_key: 0,
    unrecognized_role: 0,
    missing_content: 0,
    example_missing_assistant_message: 0,
    insufficient_examples: 0,
  };

  if (dataset.length < 10) {
    errorCounts.insufficient_examples++;
    return {
      isValid: false,
      errorCounts,
      invalidEntries: 0,
    };
  }

  let invalidEntries = 0;

  for (const entry of dataset) {
    const formatValidation = validateDataFormat(entry);

    if (!formatValidation.isValid) {
      invalidEntries++;

      // Accumulate error counts
      for (const [errorType, count] of Object.entries(
        formatValidation.errors,
      )) {
        errorCounts[errorType as FormatErrorType] += count;
      }

      continue;
    }

    // If format is valid, check message content
    const typedEntry = entry as { messages: OpenAIMessage[] };
    const messageValidation = validateMessage(
      typedEntry.messages,
      model,
      enc,
      maxTokens,
    );

    if (!messageValidation.isValid) {
      invalidEntries++;
    }
  }

  return {
    isValid: invalidEntries === 0,
    errorCounts,
    invalidEntries,
  };
}
